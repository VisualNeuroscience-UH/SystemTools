# Analysis
# from numpy import NaN
import numpy as np
import pandas as pd

# This computer git repos
from cxsystem2.core.tools import (
    change_parameter_value_in_file,
    read_config_file,
    parameter_finder,
)

# Builtins
# from argparse import ArgumentError
from pathlib import Path
import shlex
import subprocess
import os
import sys
import time
from multiprocessing import Pool
import logging
from itertools import product
import pdb
from copy import copy


class Iterator:
    def __init__(self, all_properties) -> None:
        """
        Automated buildup and execution of anatomy and physiology csv file generation, simulation and analysis.

        :param all_properties: dict, contains necessary parameters to build the iteration object. The necessary parameters (dict keys) include:
        create_csvs: flag 0, 1
        run_simulation: flag 0, 1
        run_analysis: flag 0, 1
        run_optimal_analysis, flag 0, 1
        optimal_value_delays: list, interpreted as np.linspace(list)
        parallel_analysis: bool, run simulations with subprocesses and analyze with multithreading
        analysis_type: string, either 'full_IxO' all inputs vs all outputs, or 'matched_IO' match input to correct output on diagonal
        create_log_file: bool
        iter_idx_list: list of ints or None
        input_fname_prefix: string
        input_fname_ci_suffix: string
        time_ids: dict, {name : date}
        anat_update_dict: dict including parameters to update in anatomical csv file
        phys_update_dict: dict including parameters to update in physiological csv file
        """

        validated_properties = self._validate_properties(all_properties)
        for attr, val in validated_properties.items():
            setattr(self, attr, val)

    def _validate_properties(self, all_properties):

        validated_properties = {}

        def _property_validator(attr, val, valid_vals_list, valid_types_list):
            # validated_properties is in enclosing scope, thus no need to include it to function call.
            if valid_vals_list is not None:
                if val in valid_vals_list:
                    validated_properties[attr] = val
                else:
                    raise Exception(
                        f"Parameter {attr} must one of {valid_vals_list}, aborting..."
                    )
            elif valid_types_list is not None:
                if type(val) in valid_types_list:
                    validated_properties[attr] = val
                else:
                    raise Exception(
                        f"Parameter {attr} must be of type {valid_types_list}, aborting..."
                    )

        # primarily use list of valid values. If not possible, use list of types, e.g. [int] for integer
        for attr, val in all_properties.items():
            valid_vals_list = None
            valid_types_list = None

            if attr in [
                "create_csvs",
                "run_simulation",
                "run_analysis",
                "run_optimal_analysis",
                "create_log_file",
            ]:
                valid_vals_list = [0, 1, False, True]
            elif attr in ["parallel_analysis"]:
                valid_vals_list = [True, False]
            elif attr in ["analysis_type"]:
                valid_vals_list = ["full_IxO", "matched_IO"]
            elif attr in ["iter_idx_list", "optimal_value_delays"]:
                valid_types_list = [list, type(None)]
            elif attr in ["pad_zeros"]:
                valid_types_list = [int]
                pad_format_str = "%0" + str(val) + "d"
                self.pad_format_str = pad_format_str
            elif attr in [
                "input_fname_prefix",
                "input_fname_ci_suffix",
                "optimal_value_foldername",
            ]:
                valid_types_list = [str]
            elif attr in ["time_ids", "anat_update_dict", "phys_update_dict"]:
                valid_types_list = [dict]

            _property_validator(attr, val, valid_vals_list, valid_types_list)

        return validated_properties

    def _parallel_analysis(self, this_mpi, analyzes_list, analysis_type, iter_idx_list):

        # Create parallel obj for holding _meta_fname_full, _input_file_full, and _output_folder_full.
        # The underscore denotes internally generated items, not visible in project_conf file
        midpoint, parameter, this_idx = this_mpi[:]

        # simu_title = self._get_simu_title("not_none", midpoint, parameter, this_idx)
        simu_title = self._get_simu_title(iter_idx_list, midpoint, parameter, this_idx)

        _output_folder_full = self.context.path.joinpath(simu_title)

        _input_file_full = self._get_input_filename(this_idx)
        _meta_fname_full = self._get_meta_full(_output_folder_full)
        iter_dict = {
            "_output_folder_full": _output_folder_full,
            "_input_file_full": _input_file_full,
            "_meta_fname_full": _meta_fname_full,
        }

        # Pass this obj to invoked method
        # apply in invoked functions

        if analysis_type in ["matched_IO"]:
            for this_analysis in analyzes_list:
                try:
                    self.ana.analyze_arrayrun(
                        metadata_filename=None,
                        analysis=this_analysis,
                        iter_dict=iter_dict,
                    )
                except:
                    if self.create_log_file:
                        logging.error(
                            f"\nParallel matched_IO analysis failed at midpoint {midpoint} parameter {parameter} iteration {this_idx} analysis {this_analysis}"
                        )
        elif analysis_type == "full_IxO":
            try:
                self.ana.analyze_IxO_array(
                    metadata_filename=None,
                    analyzes_list=analyzes_list,
                    iter_dict=iter_dict,
                )
            except:
                if self.create_log_file:
                    logging.error(
                        f"\nParallel IxO analysis failed at midpoint {midpoint} parameter {parameter} iteration {this_idx}"
                    )

    def _phys_updater(self, physio_config_df, param_list):

        # Find row index for correct Variable
        index = physio_config_df.index
        condition = physio_config_df["Variable"] == param_list[0]
        variable_index = index[condition].values
        assert (
            len(variable_index) == 1
        ), "Zero or nonunique variable name found, aborting..."

        if param_list[4] == "variable":
            physio_config_df.loc[variable_index[0], "Value"] = (
                param_list[2] + param_list[3]
            )
        elif param_list[4] == "key":
            condition_keys = physio_config_df["Key"] == param_list[1]
            key_indices = index[condition_keys].values
            # Find first correct Key after correct Variable. This is dangerous, because it does not check for missing Key
            key_index = key_indices[key_indices >= variable_index][0]
            physio_config_df.loc[key_index, "Value"] = param_list[2] + param_list[3]
        else:
            raise NotImplementedError(
                'Unknown row_by value, should be "key" or "variable"'
            )
        return physio_config_df

    def _csv_name_builder(self, conf_type, midpoint, conf_time_stamp, simulation_title):
        """
        Strings to build automatically conf csv file names.
        :param conf_type, str, anatomy, physiology
        :param midpoint, str, search midpoint aka template name
        :param conf_time_stamp, integer or string, conf template creation time
        :param simulation_title, string, same as sim folder name. Includes iter if relevant.
        """
        project = self.context.project

        name_part_list = []
        if "anatomy".startswith(conf_type.lower()):
            name_part_list.append("Anat")
        elif "physiology".startswith(conf_type.lower()):
            name_part_list.append("Phys")
        else:
            raise ValueError(
                f"{conf_type=}, should be anatomy, physiology, or string starting with same chars, aborting..."
            )

        name_part_list.append(project)
        name_part_list.append(midpoint)
        name_part_list.append(str(conf_time_stamp))
        name_part_list.append(simulation_title)

        csv_name = "_".join(name_part_list) + ".csv"

        return csv_name

    def _get_simu_title(self, iter_idx_list, midpoint, parameter, this_idx):

        if iter_idx_list is None:
            simu_title = f"{midpoint}_{parameter}"
        else:
            add_padding_str = self.pad_format_str % this_idx
            simu_title = f"{midpoint}_{parameter}_{add_padding_str}"  # N padded zeros 2

        return simu_title

    def _get_meta_full(self, output_folder_full):
        """
        Get metadata full path for one iterator.
        :param midpoint, str, current midpoint
        :param parameter, str, current parameter
        :param idx, integer, iterator index
        """
        metadata_list = [
            str(tp)
            for tp in output_folder_full.iterdir()
            if "metadata" in str(tp)
            and str(tp).endswith(".gz")
            and "cluster" not in str(tp)
        ]

        if len(metadata_list) == 1:
            meta_full = metadata_list[0]
        elif len(metadata_list) == 0:
            logging.error(
                f"Metadatafile not found at  {output_folder_full}, aborting..."
            )
            raise FileNotFoundError(
                f"Metadatafile not found at  {output_folder_full}, aborting..."
            )
        else:
            # get time stamps
            time_stamp_list_all = [
                fn[fn.find("_20") : fn.find(".gz")] for fn in metadata_list
            ]
            # remove possible "updated" string [f(x) if condition else g(x) for x in sequence]
            time_stamp_list = [
                ts[0 : ts.find("_updated")] if "_updated" in ts else ts
                for ts in time_stamp_list_all
            ]
            # get most recent, no matter whether it is updated or not, eg in case the updated is old relic
            meta_full = self.data_io.most_recent(
                output_folder_full, data_type="metadata"
            )
            if len(set(time_stamp_list)) == 1:
                logging.info(
                    f"More than one metadatafiles found with same time stamp. \nTook most recent metadata file: {meta_full}"
                )
            elif len(set(time_stamp_list)) > 1:
                logging.info(
                    f"More than one time stamps found in metadatafiles. \nTook most recent metadata file: {meta_full}"
                )

        return Path(meta_full)

    def _get_input_filename(self, this_idx):
        # Input file (not current injection file)
        add_padding_str = self.pad_format_str % this_idx
        input_filename = f"{self.input_fname_prefix}_{add_padding_str}.mat"
        input_file_full = self.context.input_folder.joinpath(input_filename).resolve()

        return input_file_full

    def run_iterator(self):

        path = self.context.path

        if self.create_log_file:
            timestamp = time.strftime(
                "%y%m%d_%H%M%S", time.localtime()
            )  # Simple version
            log_full_filename = Path.joinpath(path, f"Script_logfile_{timestamp}.log")
            logging.basicConfig(
                filename=log_full_filename,
                encoding="utf-8",
                level=logging.DEBUG,
                format="%(levelname)s:%(message)s",
            )

        # Go to path where conf csvs reside
        os.chdir(path)

        if self.iter_idx_list is not None:
            idx_iterator = self.iter_idx_list
        else:
            idx_iterator = [0]

        if self.coll_mpa_dict is None:
            print("No coll_mpa_dict provided, iteration disabled.")
            return
        
        # to_mpa_dict formed at the top of project_conf_module
        midpoints = self.context.to_mpa_dict["midpoints"]
        parameters = self.context.to_mpa_dict["parameters"]
        # analyzes = self.context.to_mpa_dict["analyzes"]
        analyzes = self.coll_mpa_dict["coll_ana_df"]["ana_name_prefix"]
        analyzes_list = analyzes.tolist()

        ncpus = os.cpu_count() - 1
        n_items = len(midpoints) * len(parameters) * len(idx_iterator)

        all_mpi_list = list(product(midpoints, parameters, idx_iterator))

        if self.create_csvs:
            logging.info(f"Creating {n_items} anat-phys pairs of conf csvs")
            tic = time.time()
            for this_mpi in all_mpi_list:
                midpoint, parameter, this_idx = this_mpi[:]
                conf_time_stamp = self.time_ids[midpoint]

                # Anatomy and system
                anat_config_template = self._csv_name_builder(
                    "anat", midpoint, conf_time_stamp, "midpoint"
                )

                simu_title = self._get_simu_title(
                    self.iter_idx_list, midpoint, parameter, this_idx
                )

                anat_config_iter = self._csv_name_builder(
                    "anat", midpoint, conf_time_stamp, simu_title
                )
                change_parameter_value_in_file(
                    anat_config_template,
                    anat_config_iter,
                    "simulation_title",
                    simu_title,
                )

                for this_key in self.anat_update_dict.keys():
                    change_parameter_value_in_file(
                        anat_config_iter,
                        anat_config_iter,
                        this_key,
                        self.anat_update_dict[this_key],
                    )

                # Physiology
                phys_config_template = self._csv_name_builder(
                    "phys", midpoint, conf_time_stamp, "midpoint"
                )
                physio_config_df = read_config_file(phys_config_template, header=True)
                phys_config_iter = self._csv_name_builder(
                    "phys", midpoint, conf_time_stamp, simu_title
                )

                for param_list in self.phys_update_dict["current_injection"]:
                    # We update the ci_filename to contain the iteration index if multiple noise files requested.
                    if (
                        param_list[0] == "ci_filename"
                        and self.iter_idx_list is not None
                    ):
                        add_padding_str = self.pad_format_str % this_idx
                        param_list[
                            2
                        ] = f"'{self.input_fname_prefix}_{add_padding_str}{self.input_fname_ci_suffix}'"
                        physio_config_df = self._phys_updater(
                            physio_config_df, param_list
                        )
                    else:
                        physio_config_df = self._phys_updater(
                            physio_config_df, param_list
                        )
                for param_list in self.phys_update_dict[parameter]:
                    physio_config_df = self._phys_updater(physio_config_df, param_list)
                physio_config_df.to_csv(phys_config_iter, index=False, header=True)
            toc = time.time()
            logging.info(f"\Creating csvs took {(toc-tic):.2f} seconds")

        if self.run_simulation:
            logging.info(f"Running simulations with {n_items} processes")
            tic = time.time()
            proc_list = []
            proc_count = 0

            for this_mpi in all_mpi_list:
                midpoint, parameter, this_idx = this_mpi[:]
                conf_time_stamp = self.time_ids[midpoint]

                conf_time_stamp = self.time_ids[midpoint]
                if self.iter_idx_list is None:
                    simu_title = f"{midpoint}_{parameter}"
                else:
                    add_padding_str = self.pad_format_str % this_idx
                    simu_title = f"{midpoint}_{parameter}_{add_padding_str}"

                anat_config_iter = self._csv_name_builder(
                    "anat", midpoint, conf_time_stamp, simu_title
                )
                phys_config_iter = self._csv_name_builder(
                    "phys", midpoint, conf_time_stamp, simu_title
                )

                funcion_call_str = (
                    f"cxsystem2 -a {anat_config_iter} -p {phys_config_iter}"
                )
                logging.info(funcion_call_str)
                funcion_call = shlex.split(funcion_call_str)
                proc = subprocess.Popen(funcion_call)
                proc_list.append(proc)
                proc_count += 1

                # Necessary for distinct time stamps in subprocesses
                sys.stdout.flush()  # print buffer flush for real pause
                time.sleep(1.1)

                if len(proc_list) == ncpus or proc_count == n_items:
                    # Wait for the end of all processes for each process
                    for subproc in proc_list:
                        logging.info(f"\nsubproc time = {time.time() - tic}")
                        subproc.wait()
                    proc_list = []
            toc = time.time()
            logging.info(f"\nSimulations took {(toc-tic):.2f} seconds")

        if self.run_analysis:
            # Create iterator including all midpoints, parameters and iterations. Note that the order is critical for correct names
            logging.info(
                f"Running analyses {analyzes_list}, for {n_items} simulation results data file"
            )
            tic = time.time()
            pass_flag = 0
            _tmp_all_mpi_list = copy(all_mpi_list)
            while pass_flag == 0:

                if self.parallel_analysis:
                    pool = Pool(ncpus)
                    for this_mpi in _tmp_all_mpi_list:
                        # Run multiprocessing. In windows this requires the if __name__ == 'main': guard in the project_conf_module.
                        res = pool.apply_async(
                            self._parallel_analysis,
                            (
                                this_mpi,
                                analyzes_list,
                                self.analysis_type,
                                self.iter_idx_list,
                            ),
                        )

                    pool.close()
                    pool.join()
                else:
                    for this_mpi in _tmp_all_mpi_list:
                        ## With no multiprocessing you call directly
                        self._parallel_analysis(
                            this_mpi,
                            analyzes_list,
                            self.analysis_type,
                            self.iter_idx_list,
                        )

                failed_grcaus_iterations = []
                # Search failed files
                for this_mpi in _tmp_all_mpi_list:
                    midpoint, parameter, this_idx = this_mpi[:]
                    add_padding_str = self.pad_format_str % this_idx
                    self.context.output_folder = Path(
                        f"{midpoint}_{parameter}_{add_padding_str}"
                    )
                    tmp_full_filename = Path.joinpath(
                        self.context.output_folder,
                        "_tmp_regression_errors.txt",
                    )
                    reg_error_filename = Path.joinpath(
                        self.context.output_folder,
                        "regression_errors.txt",
                    )
                    if tmp_full_filename.is_file():
                        failed_grcaus_iterations.append(this_mpi)
                        # read tmp file and append it to regression errors txt
                        f1 = open(reg_error_filename, "a+")
                        f2 = open(tmp_full_filename, "r")
                        f1.write(f2.read())
                        f1.close()
                        f2.close()
                        Path.unlink(tmp_full_filename)

                # Set failed files for re-iteration
                if failed_grcaus_iterations:
                    if self.create_log_file:
                        logging.error(
                            f"\nGrCaus failed at mpi:s {failed_grcaus_iterations}"
                        )
                    _tmp_all_mpi_list = failed_grcaus_iterations
                else:
                    pass_flag = 1

            toc = time.time()
            if self.create_log_file:
                logging.info(f"\nAnalysis took {(toc-tic):.2f} seconds")

        if self.run_optimal_analysis:
            # Here, we exceptionally use full analysis names, not the shortened ones.
            # This is because the analysis names are used one-by-one in the analysis

            analyzes_list = self.context.to_mpa_dict["analyzes"]

            logging.info(
                f"Running optimal values for analyze methods {analyzes_list}, for {len(idx_iterator)} inputs"
            )
            tic = time.time()

            # Check existing folder at self.context.path for optimal values. If not, create it.
            if not self.context.path.joinpath(self.optimal_value_foldername).is_dir():
                self.context.path.joinpath(self.optimal_value_foldername).mkdir()

                # get inputs. Loop over all inputs. Put values to dict.
            input_dict = {}
            for this_idx in idx_iterator:
                input_filename = self._get_input_filename(this_idx)
                analog_input = self.data_io.get_data(input_filename, data_type=None)
                input_dict[this_idx] = analog_input
            delays = self.optimal_value_delays

            # dt = self.context.dt
            dt = 0.0001

            delay_in_ms = (
                np.floor(np.linspace(delays[0], delays[1], delays[2])).astype(int)
                * dt
                * 1000
            )

            optimal_values_mtx = (
                np.zeros(
                    (
                        len(analyzes_list),
                        len(delay_in_ms),
                        len(idx_iterator),
                    )
                )
                * np.nan
            )
            nonoptimal_values_mtx = (
                np.zeros(
                    (
                        len(analyzes_list),
                        len(delay_in_ms),
                        len(idx_iterator),
                    )
                )
                * np.nan
            )

            # Map the analysis names to coll_mpa_dict["coll_ana_df"]["csv_col_name"]
            analyzes_col_name_list = []
            for this_ana in analyzes_list:
                analyzes_col_name_list.append(
                    self.coll_mpa_dict["coll_ana_df"].loc[this_ana]["csv_col_name"]
                )

            optimal_values_df = pd.DataFrame(
                columns=analyzes_col_name_list, index=delay_in_ms
            )
            nonoptimal_values_df = pd.DataFrame(
                columns=analyzes_col_name_list, index=delay_in_ms
            )

            # Duplicate the column names in the dataframes with
            # suffixes "_mean" and "_SD".
            for col in optimal_values_df.columns:
                optimal_values_df[col + "_mean"] = optimal_values_df[col]
                optimal_values_df[col + "_SD"] = optimal_values_df[col]
                nonoptimal_values_df[col + "_mean"] = nonoptimal_values_df[col]
                nonoptimal_values_df[col + "_SD"] = nonoptimal_values_df[col]
            # Remove original col names from dataframes.
            optimal_values_df.drop(columns=analyzes_col_name_list, inplace=True)
            nonoptimal_values_df.drop(columns=analyzes_col_name_list, inplace=True)

            for this_ana_idx, this_ana in enumerate(analyzes_list):
                for this_idx in idx_iterator:
                    data_for_mtx = self.ana.optimal_value_analysis(
                        analyze=this_ana,
                        delay=delays,
                        analog_input=input_dict[this_idx],
                    )
                    (
                        (delay_in_ms, value_array, noise_array, foo),
                        foo1,
                        foo2,
                    ) = data_for_mtx

                    optimal_values_mtx[this_ana_idx, :, this_idx] = value_array
                    nonoptimal_values_mtx[this_ana_idx, :, this_idx] = noise_array

            # Take nanmean and nansd from the mtx. Set values to dataframes.
            for this_ana_idx, this_ana_col in enumerate(analyzes_col_name_list):
                optimal_values_df[this_ana_col + "_mean"] = np.nanmean(
                    optimal_values_mtx[this_ana_idx, :, :], axis=1
                )
                optimal_values_df[this_ana_col + "_SD"] = np.nanstd(
                    optimal_values_mtx[this_ana_idx, :, :], axis=1
                )
                nonoptimal_values_df[this_ana_col + "_mean"] = np.nanmean(
                    nonoptimal_values_mtx[this_ana_idx, :, :], axis=1
                )
                nonoptimal_values_df[this_ana_col + "_SD"] = np.nanstd(
                    nonoptimal_values_mtx[this_ana_idx, :, :], axis=1
                )

                logging.info(f"...done for {this_ana_col}.")
                print(f"...done for {this_ana_col}.")
            print(
                f"All optimal analyses done, saving to {self.optimal_value_foldername}"
            )

            # save numpy matrix to gzipped pickle file
            filename = Path.joinpath(
                self.context.path.joinpath(self.optimal_value_foldername),
                f"optimal_value_analysis_{self.context.project}.gz",
            )
            optimal_values_dict = {}
            optimal_values_dict["optimal_values"] = optimal_values_mtx
            optimal_values_dict["nonoptimal_values"] = nonoptimal_values_mtx
            optimal_values_dict["optimal_values_df"] = optimal_values_df
            optimal_values_dict["nonoptimal_values_df"] = nonoptimal_values_df
            self.data_io.write_to_file(filename, optimal_values_dict)

            # save dataframes to csv
            filename = Path.joinpath(
                self.context.path.joinpath(self.optimal_value_foldername),
                f"optimal_values_df_{self.context.project}.csv",
            )
            optimal_values_df.to_csv(filename, index=True, index_label="delay_in_ms")

            filename = Path.joinpath(
                self.context.path.joinpath(self.optimal_value_foldername),
                f"nonoptimal_values_df_{self.context.project}.csv",
            )
            nonoptimal_values_df.to_csv(filename, index=True, index_label="delay_in_ms")
