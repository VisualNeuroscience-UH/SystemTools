# This computer git repos
from project.project_base_module import ProjectBase
from project.project_utilities_module import ProjectUtilities
from context.context_module import Context
from data_io.data_io_module import DataIO
from construction.connection_translation_module import ConnectionTranslator
from analysis.analysis_module import Analysis
from viz.viz_module import Viz
from analysis.cx_parser_module import CXParser
from iterate.iterate_module import Iterator
from analysis.statistics_module import Statistics

# Builtin
import pdb
import time
import shlex
import subprocess

# Analysis
import pandas as pd


"""
Module on project-specific data analysis.
This configures analysis. 
Specific methods are called at the bottom after the if __name__=='__main__':

Simo Vanni 2021
"""


class ProjectManager(ProjectBase, ProjectUtilities):

    # __slots__ = ('__dict__') # Add here constants later

    map_ana_names = {
        "meanfr": "MeanFR",
        "eicurrentdiff": "EICurrentDiff",
        "grcaus": "GrCaus",
        "meanvm": "MeanVm",
        "coherence": "Coherence",
        "normerror": "NormError",
        "classify": "Classify",
        "transferentropy": "TransferEntropy",
        "accuracy": "accuracy",
        "_p": "_p",
        "tedrift": "TEDrift",
        "edist": "EDist",
    }

    map_ixo_names = {
        "grcaus": "GrCaus",
        "coherence": "Coherence",
        "normerror": "NormError",
        "transferentropy": "TransferEntropy",
    }

    map_data_types = {
        "meanfr": "spikes_all",
        "eicurrentdiff": "vm_all",
        "transferentropy": "vm_all",
        "grcaus": "vm_all",
        "meanvm": "vm_all",
        "coherence": "vm_all",
        "normerror": "vm_all",
        "classify": "vm_all",
        "meanerror": "vm_all",
        "edist": "vm_all",
    }

    map_stat_types = {
        "Coherence": {"Mean": "mean", "Latency": "mean"},
        "TransferEntropy": {"TransfEntropy": "mean"},
        "GrCaus": {
            "Information": "mean",
            "p": "mean",
            "Latency": "mean",
            "FitQuality": "mean",
            "InfoAsTE": "mean",
        },
        "NormError": {"ExcErr": "mean", "InhErr": "mean", "SimErr": "mean"},
        "TEDrift": {"TEDrift": "mean"},
        "EDist": {"EDist": "mean"},
    }

    TE_args = {
        "max_time_lag_seconds": 0.1,
        "downsampling_factor": 40,
        "n_states": 4,  # 4, 2
        "embedding_vector": 1,
        "te_shift_start_time": 0.004,  # only for TE_Drift analysis
        "te_shift_end_time": 0.08,
    }  # only for TEDrift analysis
    GrCaus_args = {
        "max_time_lag_seconds": 0.1,
        "downsampling_factor": 40,  # 40, 2
        "save_gc_fit_dg_and_QA": False,
        "show_gc_fit_diagnostics_figure": False,
    }
    NormError_args = {"decoding_method": "least_squares", "do_only_simerror": True}
    kw_ana_args = {
        "TE_args": TE_args,
        "GrCaus_args": GrCaus_args,
        "NormError_args": NormError_args,
    }

    def __init__(self, **all_properties):
        """
        Main project manager.
        In init we construct other classes and inject necessary dependencies. This class is allowed to house project-dependent data and methods.
        """

        # ProjectManager is facade to Context.
        context = Context(all_properties)

        # Get correct context attributes. Empty properties return all existing project attributes to context. That is what we want for the project manager
        self.context = context.set_context()

        data_io = DataIO(context, map_ana_names=self.map_ana_names)
        self.data_io = data_io

        cxparser = CXParser()
        self.cxparser = cxparser

        self.ct = ConnectionTranslator(context, data_io)

        # Build necessary analyzes from to_spa_dict
        self.build_mid_par_ana()
        stat_tests = Statistics()

        ana = Analysis(
            # Interfaces
            context,
            data_io,
            cxparser,
            stat_tests,
            # Dictionaries
            map_ana_names=self.map_ana_names,
            map_ixo_names=self.map_ixo_names,
            map_data_types=self.map_data_types,
            map_stat_types=self.map_stat_types,
            extra_ana_args=self.kw_ana_args,
            coll_spa_dict=self.coll_spa_dict,
            # Methods, which are needed also elsewhere
            round_to_n_significant=self.round_to_n_significant,
            pp_df_full=self.pp_df_full,
            end2idx=self.end2idx,
            update_metadata=self.update_metadata,
        )

        self.ana = ana

        self.viz = Viz(
            # Interfaces
            context,
            data_io,
            ana,
            cxparser,
            # Dictionaries
            map_ana_names=self.map_ana_names,
            coll_spa_dict=self.coll_spa_dict,
            # Methods, which are needed also elsewhere
            round_to_n_significant=self.round_to_n_significant,
            end2idx=self.end2idx,
        )

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        if isinstance(value, Context):
            self._context = value
        else:
            raise AttributeError(
                "Trying to set improper context. Context must be a context object."
            )

    @property
    def data_io(self):
        return self._data_io

    @data_io.setter
    def data_io(self, value):
        if isinstance(value, DataIO):
            self._data_io = value
        else:
            raise AttributeError(
                "Trying to set improper data_io. Data_io must be a DataIO object."
            )

    def build_mid_par_ana(self):
        """
        Build mapping from to_spa_dict in conf file to dataframes which include parameter and analysis details.
        """
        if not hasattr(self.context, "to_spa_dict") or self.context.to_spa_dict is None:
            self.coll_spa_dict = None
            return

        startpoints = self.context.to_spa_dict["startpoints"]
        parameters = self.context.to_spa_dict["parameters"]
        analyzes = self.context.to_spa_dict["analyzes"]

        all_startpoints_list = ["Comrad", "Bacon", "HiFi"]

        all_param_col_list = ["Name", "Unit", "Dims"]
        all_param_dict = {
            "C": ["Capacitance", "pF", "2d"],
            "gL": ["Leak conductance", "nS", "2d"],
            "VT": ["AP threshold", "mV", "2d"],
            "EL": ["Leak equilibrium potential", "mV", "2d"],
            "delay": ["Synaptic delay", "ms", "1d"],
            "tau_e": ["AMPA time constant", "ms", "1d"],
            "tau_i": ["GABA-A time constant", "ms", "1d"],
        }
        all_param_df = pd.DataFrame.from_dict(
            all_param_dict, orient="index", columns=all_param_col_list
        )

        if parameters is not None:
            coll_param_df = pd.DataFrame(all_param_df, index=parameters)

        all_ana_col_list = ["csv_col_name", "best_is", "ana_name_prefix"]
        all_ana_dict = {
            "Simulation Error": ["NormError_SimErr", "min", "NormError"],
            "Transfer Entropy": [
                "TransferEntropy_NG3_L4_SS_L4_TransfEntropy",
                "max",
                "TransferEntropy",
            ],
            "Transfer Entropy Latency": [
                "TransferEntropy_NG3_L4_SS_L4_Latency",
                "min",
                "TransferEntropy",
            ],
            "Granger Causality": ["GrCaus_NG3_L4_SS_L4_Information", "max", "GrCaus"],
            # "Granger Causality": ["GrCaus_NG3_L4_SS_L4_GCAsTE", "max", "GrCaus"],
            "GC as TE": ["GrCaus_NG3_L4_SS_L4_GCAsTE", "max", "GrCaus"],
            "Coherence": ["Coherence_NG3_L4_SS_L4_Mean", "max", "Coherence"],
            "Excitatory Firing Rate": ["MeanFR_NG1_L4_CI_SS_L4", "min", "MeanFR"],
            "Inhibitory Firing Rate": ["MeanFR_NG2_L4_CI_BC_L4", "min", "MeanFR"],
            "Euclidean Distance": ["EDist_NG3_L4_SS_L4_EDist", "min", "EDist"],
        }

        all_ana_df = pd.DataFrame.from_dict(
            all_ana_dict, orient="index", columns=all_ana_col_list
        )

        if analyzes is not None:
            coll_ana_df = pd.DataFrame(all_ana_df, index=analyzes)

        # Assertions
        if startpoints is not None:
            coll_start_list = list(sorted(set(all_startpoints_list) & set(startpoints)))
            assert (
                len(coll_start_list) > 0
            ), 'variable "startpoints" not set or names do not match get_collated_analyzes method data'
        if parameters is not None:
            collated_parameters = list(set(all_param_dict.keys()) & set(parameters))
            assert (
                len(collated_parameters) > 0
            ), 'variable "parameters" not set or names do not match get_collated_analyzes method data'
        if analyzes is not None:
            coll_ana_list = list(set(all_ana_dict.keys()) & set(analyzes))
            assert (
                len(coll_ana_list) > 0
            ), 'variable "analyzes" not set or names do not match get_collated_analyzes method data'

        coll_spa_dict = {}
        if parameters is not None:
            coll_spa_dict["coll_param_df"] = coll_param_df
        if startpoints is not None:
            coll_spa_dict["coll_start_list"] = coll_start_list
        if analyzes is not None:
            coll_spa_dict["coll_ana_df"] = coll_ana_df

        self.coll_spa_dict = coll_spa_dict

    def build_iterator(self, **all_parameters):
        """
        Constructs iterator object. This is called from project configuration file.
        This is similar to __init__ but allows for the iterator to be built after the project manager is instantiated.
        """
        IT = Iterator(all_parameters)
        IT.context = self.context
        IT.data_io = self.data_io
        IT.ana = self.ana
        IT.coll_spa_dict = self.coll_spa_dict
        IT.phys_updater = self.phys_updater  # From project utilities
        self.IT = IT

    def run_iterator(self):
        '''
        Runs iterator object.
        '''
        if hasattr(self, "IT"):
            self.IT.run_iterator()
        else:
            print("Iterator not built, not iterating...")


# if __name__=='__main__':
