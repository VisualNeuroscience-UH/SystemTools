# Visualization
import matplotlib.pyplot as plt

# Builtin
from pathlib import Path
import sys
from math import nan
import pdb

# This computer git repos
from project.project_manager_module import ProjectManager

"""
Use keyword substring "file" in filenames, and "folder" in foldernames to assert that they are turned into pathlib objects. Path structure is assumed to be root_path/project/experiment/output_folder

Abbreviations:
ana : analysis
ci : current injection
col : column
coll : collated, collected
conn : connections
full : full absolute path
mid : midpoint
param : parameter
"""

"""
Main paths in different operating systems
"""
if sys.platform == "linux":
    root_path = "/opt3/Laskenta/Models"  # pikkuveli
    # root_path = "/opt2/Laskenta_ssd/Models"  # isosisko
    # root_path = "/opt3/Matteo/"  # Matteo
elif sys.platform == "win32":
    root_path = r"C:\Users\Simo\Laskenta\Models"


"""
Project name
"""
project = "Deneve"  # VenDor Deneve

"""
Current experiment
"""
experiment = "Single_narrow_iteration_R1"  # 'Canonical_folder',sandbox, Single_narrow_iterations, classification_1D exp_iter10 Single_narrow_iterations310
# experiment = "Canonical_folder"  # 'Canonical_folder',sandbox, Single_narrow_iterations, classification_1D exp_iter10 Single_narrow_iterations310

"""
### Housekeeping ###. Do not comment out.
"""
path = Path.joinpath(Path(root_path), Path(project), experiment)


"""
Input context
"""
input_folder = "../in"
# matlab_workspace_file = "workspace_220104_fast.mat"
matlab_workspace_file = "workspace_deneve_SingleSpike.mat"
# conn_skeleton_file_in = "skeleton_connections_20220210_2242135.gz"
conn_skeleton_file_in = "Replica_skeleton_connections_20210211_1453238_L4_SS.gz"
# conn_file_out = "connections_learned_6by6_ci.gz"
conn_file_out = "connections_SingleSpike_ci.gz"
input_filename = "noise_210406_SingleSpike.mat"  # "noise_210406_SingleSpike.mat"  # "input_quadratic_oscillation_220215.mat" "input_noise_220215.mat"

"""
Data context for single files and arrays. These midpoint and parameter strings are used only in this module.
"""
midpoint = "HiFi"  # HiFi, Comrad, Bacon
parameter = "midpoint"

output_folder = f"{midpoint}_{parameter}"

"""
Analysis context
t_idx_start cuts the beginning and t_idx_end the end of time samples. Use 0, None or integer btw [0 N_time_samples]. If negative, will count from the end.
"""
t_idx_start = 2000
t_idx_end = -2000
NG_output = "NG3_L4_SS_L4"

"""
Data context for multiple analyzes and visualizations. 
midpoints: 'Comrad', 'Bacon', 'HiFi'
parameters: 'C', 'gL', 'VT', 'EL', 'delay'
analyzes: 
    'Coherence', 'Granger Causality', 'GC as TE', 'Transfer Entropy', 'Simulation Error', 'Excitatory Firing Rate', 'Inhibitory Firing Rate', 'Euclidean Distance'
If you give to_mpa_dict = None, only single files will be handled.
"""
to_mpa_dict = {
    "midpoints": ["Comrad", "Bacon", "HiFi"],
    # "midpoints": ["Comrad", "Bacon"],
    "parameters": ["C", "gL", "VT", "EL", "delay"],
    # "parameters": ["gL"],
    "analyzes": [
        # "Inhibitory Firing Rate",
        "Coherence",
        "Granger Causality",
        "Transfer Entropy",
        "Simulation Error",
        # 'GC as TE',
        # "Excitatory Firing Rate",
        # "Euclidean Distance",
    ],
}

profile = False

if __name__ == "__main__":

    if profile is True:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    """
    ### Housekeeping ###. Do not comment out.
    """
    PM = ProjectManager(
        path=path,
        input_folder=input_folder,
        output_folder=output_folder,
        matlab_workspace_file=matlab_workspace_file,
        conn_skeleton_file_in=conn_skeleton_file_in,
        conn_file_out=conn_file_out,
        input_filename=input_filename,
        NG_name=NG_output,
        t_idx_start=t_idx_start,
        t_idx_end=t_idx_end,
        to_mpa_dict=to_mpa_dict,
        project=project,
        experiment=experiment,
    )

    #################################
    ### Project files & Utilities ###
    #################################

    """
    Transforms Deneve's simulation connections from .mat file to CxSystem .gz format.
    Creates conn_file_out to input folder
    """
    # PM.ct.deneve_replace_conn(show_histograms=False, constant_scaling=True, constant_value=1e-9)

    """
    Randomize learned connectivity for control conditions. Optional conditions 
    'EI' all mutual NG1 and NG2 connections
    'D' from EI to output
    'ALL' all of the above
    """
    # PM.deneve_create_control_conn(conn='D')

    """
    Creates file named input_filename but with _ci.mat suffix to experiment folder
    Condition 'ALL' randomizes connections from input to E--for those connections set randomize=True
    For iterations use the loop
    """
    # # PM.ct.deneve_create_current_injection(randomize=False)
    # for idx in range(1,101):
    #     PM.context.input_filename = Path(f'freq_{idx:02}.mat')
    #     PM.ct.deneve_create_current_injection(randomize=False)

    """
    Show connections
    """
    # PM.viz.show_conn(conn_file='connections_SingleSpike_ci.gz', hist_from='L4_CI_SS_L4__to__L4_CI_SS_L4_soma', savefigname='')

    """
    Print metadata
    Use data_type='metadata' with no filename, if you want the most recent file in the current folder.
    """
    # meta_fname='metadata__20220615_1256589.gz'
    # metadata_df = PM.data_io.get_data(filename=meta_fname, data_type='metadata')
    # PM.pp_df_full(metadata_df)

    """
    Compile cluster run metadata parts and transfer data to project folder
    """
    # metapath=r'/opt3/Laskenta/Models/Deneve/HPC_narrow_test/cluster_run_20211007_1549467/cluster_metadata_20211007_1549467.pkl'
    # PM.cluster_metadata_compiler_and_data_transfer(metapath)
    # PM.multiple_cluster_metadata_compiler_and_data_transfer()

    """
    Map workspaces between operating systems. Note that this assumes you are calling the metadatafile from the new workspace.
    """
    # PM.cluster_to_ws_metadata_mapper(metapath)

    """
    Optimal value analysis. For each input, calculates information transfer metrics for itself. Delay in timepoints. Same preprocessing applied as in the data analysis.
    Analyzes: 'Coherence', 'Granger Causality', 'Transfer Entropy', 'Simulation Error', 'GC as TE'
    delay: integer (delay in timepoints) or list. The list will be interpreted as np.linspace(list), e.g. list=[0,1000,20] are the start, stop and number of items.
    
    ### Housekeeping ###. Do not comment out the next two lines.
    """
    optimal_value_foldername = "optimal_values_EDist"
    optimal_description_name = "optimal_unfit_description_EDist.csv"

    # delay = [0,2000,40] # [0, 100, 10] # [0, 1000, 1000] [0,1000,20], 50 [0,200,100]
    # data_for_viz = PM.ana.optimal_value_analysis(analyze='Granger Causality', delay=delay)
    # PM.viz.show_optimal_value_analysis(data_for_viz, savefigname='')

    # folderpath = Path.joinpath(Path(path), optimal_value_foldername)
    # PM.viz.show_iter_optimal_value_analysis(folderpath, savefigname='')

    # PM.ana.describe_optimal_values(folderpath=Path.joinpath(Path(path), optimal_value_foldername), savename=optimal_description_name)

    # ####################################################
    # ###### Do not use unless you know what you are doing
    # ####################################################

    # ## Destroy data from .gz dict
    # dict_key_list={'vm_all' : ['NG1_L4_CI_SS_L4', 'NG2_L4_CI_BC_L4']}
    # PM.destroy_from_folders(path=path, dict_key_list=dict_key_list)

    # ## Manipulate metadata.gz manually
    # replace_dict = {
    #     "columns": ["Full path", "Dimension-1 Value"],
    #     "rows": [1, 3],
    #     "find": ["_C300pF_C", "30.0"],
    #     "replace": ["_C400pF_C", "40.0"],
    # }
    # meta_full = Path(r'C:\Users\Simo\Laskenta\Git_Repos\SystemTools\tests\out\Comrad_C_0\metadata__20211008_1039431.gz')
    # PM.metadata_manipulator(
    #     meta_full=meta_full, multiply_rows=2, replace_dict=replace_dict
    # )

    # ##########################################
    # ###### Analysis & Viz, single files ######
    # ##########################################

    """
    Show input and data. If  file_to_display = None, the function selects the most recent data file in output_folder.
    """
    # file_to_display = r"/opt3/Laskenta/Models/Deneve/adex_test/test0/test0_results_20221115_1813386.gz"  # None
    file_to_display = None

    # PM.viz.show_readout_on_input(results_filename=file_to_display, normalize=False, unit_idx_list=[0], savefigname='')
    # Best possible outputsignal, given the leaky spikes in readout group
    # PM.viz.show_estimate_on_input(results_filename=None, simulation_engine='cxsystem', readout_group='E', unit_idx_list=[5]) # Simulation engines: 'cxsystem','matlab'. Matlab needs filename. Readout_groups 'E','I'.
    # PM.viz.show_input_to_readout_coherence(results_filename=file_to_display, savefigname='',signal_pair=[0,0]) # HiFi_C0_I180E70_Coh.svg

    """
    Show spikes and continuous data such as Vm. Note that param name can be any used dynamic variable in Brian2 equations
    """
    # PM.viz.show_spikes(results_filename=file_to_display, savefigname="")

    # # neuron_index = {"NG3_L4_SS_L4": [0, 1, 2]}
    # # neuron_index = {"NG1_L4_CI_SS_L4": [0, 1, 2]} # Only three neurons are monitored
    # neuron_index = {"NG2_L4_CI_BC_L4": [0, 1, 2]} # Only three neurons are monitored
    # PM.viz.show_analog_results(
    #     results_filename=None,
    #     savefigname="",
    #     param_name="vm",
    #     startswith="NG2",
    #     neuron_index=neuron_index,
    # )

    # PM.viz.show_analog_results(
    #     results_filename=None,
    #     savefigname="",
    #     param_name="w",
    #     startswith="NG2",
    #     neuron_index=neuron_index,
    # )

    """
    Show E and I currents: Not measured in VenDor
    """
    # neuron_index = {'NG1_L4_CI_SS_L4' : 150, 'NG2_L4_CI_BC_L4' : 37, 'NG3_L4_SS_L4' : 1} # None
    # PM.show_currents(results_filename=None, savefigname='', neuron_index=neuron_index)

    ##########################################################################
    ###### Analysis & Viz, array runs, single midpoint, single analysis ######
    ##########################################################################

    """
    If the following is active, the displayed array analysis figures are saved as 
    arrayIdentifier_analysis_identifier.svg at your path.
    """
    # PM.viz.save_figure_with_arrayidentifier = f'{midpoint}_XXX'
    # PM.viz.save_figure_to_folder = f'Analysis_Figures'

    """
    Calculate full array, i.e. all input-output pairs for requested analyzes. This is used for classification performance
    as well as for getting the mean/median value. See method text for details.
    """
    # PM.ana.analyze_IxO_array(metadata_filename=None, analyzes_list=['NormError', 'Coherence', 'TransferEntropy', 'GrCaus'])

    """
    Analyze and show a single arrayrun.
    """
    # meta_fname = "metadata__20220615_1256589.gz"
    # PM.ana.analyze_arrayrun(metadata_filename=meta_fname, analysis='NormError')
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=None, analysis="NormError", variable_unit="a.u."
    # )

    # PM.ana.analyze_arrayrun(metadata_filename=meta_fname, analysis='MeanFR')
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=None, analysis="MeanFR", variable_unit="Hz", NG_id_list=["NG1"]
    # )  # Empty NG_id_list for all groups

    # PM.ana.analyze_arrayrun(metadata_filename=meta_fname, analysis='Coherence')
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=None, analysis="Coherence", NG_id_list=["NG3"]
    # )

    # PM.ana.analyze_arrayrun(metadata_filename=meta_fname, analysis='TransferEntropy')
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=None, analysis="TransferEntropy", NG_id_list=["NG3"]
    # )

    # PM.ana.analyze_arrayrun(metadata_filename=meta_fname, analysis='GrCaus')
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=None, analysis="GrCaus", NG_id_list=["NG3"]
    # )

    # PM.ana.analyze_arrayrun(metadata_filename=meta_fname, analysis='EDist')
    # PM.viz.show_analyzed_arrayrun(csv_filename=None, analysis='EDist')

    #######################################################
    ###### Viz, single array runs, multiple analyzes ######
    #######################################################

    """
    The system_polar_bar method operates on output folder and its subfolders. It searches for csv files; only valid array 
    analysis csv files are collected. 
    """
    # PM.viz.system_polar_bar(row_selection = [2, 22, 45], folder_name = None)

    ############################################################
    ###### Analyze & Viz, array runs, multiple iterations ######
    ############################################################

    """
    Save any figure below to path/output_path/save_figure_to_folder
    """
    # PM.viz.save_figure_with_arrayidentifier = f'Fig7B_6x6'
    # PM.viz.save_figure_to_folder = f'Analysis_Figures'
    # plt.rcParams['figure.figsize'] = (16, 4) #Saved figure size

    """
    Visualize the mean of multiple iterations in 2D and 3D
    E.g. 'MeanFR', 'Coherence', 'GrCaus', 'NormError', 'TransferEntropy'
    """
    # analysis = "MeanFR"
    # # csv_filename = "MeanFR_mean.csv"
    # csv_filename = "MeanFR_mean.csv"
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=csv_filename,
    #     analysis=analysis,
    #     variable_unit="Hz",
    #     logscale=True,
    #     annotation_2D=False,
    # )

    # analysis = 'EDist'
    # csv_filename = 'EDist_MeanFR_mean.csv'
    # PM.viz.show_analyzed_arrayrun(csv_filename=csv_filename, analysis=analysis, variable_unit='a.u.', logscale=True, annotation_2D=False)

    # analysis = "NormError"
    # csv_filename = "Coherence_GrCaus_NormError_TransferEntropy_mean.csv"
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=csv_filename,
    #     analysis=analysis,
    #     variable_unit="Normalized error",
    #     annotation_2D=False,
    # )

    # analysis = "Coherence"
    # csv_filename = "Coherence_GrCaus_NormError_TransferEntropy_mean.csv"
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=csv_filename,
    #     analysis=analysis,
    #     variable_unit="Sum a.u.",
    #     annotation_2D=False,
    # )

    # analysis = "GrCaus"
    # csv_filename = "Coherence_GrCaus_NormError_TransferEntropy_mean.csv"
    # # csv_filename = 'GrCaus_mean.csv'
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=csv_filename,
    #     analysis=analysis,
    #     variable_unit="bit",
    #     annotation_2D=False,
    # )

    # analysis = "TransferEntropy"
    # csv_filename = "Coherence_GrCaus_NormError_TransferEntropy_mean.csv"
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=csv_filename,
    #     analysis=analysis,
    #     variable_unit="bit/time point",
    #     annotation_2D=False,
    # )

    """
    Visualize classification accuracy score in 2D and 3D
    """
    # analysis = "accuracy"  # 'accuracy', '_p'
    # csv_filename = "Coherence_GrCaus_NormError_TransferEntropy_IxO_accuracy.csv"
    # PM.viz.show_analyzed_arrayrun(
    #     csv_filename=csv_filename,
    #     analysis=analysis,
    #     variable_unit="a.u.",
    #     annotate_with_p=True,
    # )

    """
    Show xy plot allows any parametric data plotted against each other.
    Uses seaborn regplot or lineplot. Seaborn options easy to include into code (viz_module).
    All analyzes MUST be included into to_mpa_dict
    Same data at the x and y axis simultaneously can be used with regplot.
    If compiled_type is accuracy, and only mean datatype is available, 
    uses the mean. 

    midpoints: 'Comrad', 'HiFi', 'Bacon'
    parameters: 'C', 'gL', 'VT', 'EL', 'delay'
    analyzes: 
    'Coherence', 'Granger Causality', 'Transfer Entropy', 'Simulation Error', 'Excitatory Firing Rate', 'Inhibitory Firing Rate', 'Euclidean Distance'

    kind: regplot, binned_lineplot 
        regplot is scatterplot, where only single midpoint and parameter should be plotted at a time. draw_regression available.
        binned_lineplot bins x-data, then compiles parameters/midpoints and finally shows distinct midpoints/parameters (according to "hue") with distinct hues. Error shading 
        indicates 95% confidence interval, obtained by bootstrapping the data 1000 times (seaborn default)
    """
    # xy_plot_dict = {
    #     "x_ana": [
    #         # "Euclidean Distance",
    #         # "Transfer Entropy",
    #         "Excitatory Firing Rate",
    #     ],  # multiple allowed => subplot rows, unless ave
    #     "x_mid": [
    #         "Comrad",
    #         "Bacon",
    #         # 'HiFi',
    #     ],  # single allowed, multiple (same mids for y) if type binned_lineplot
    #     # "x_para": ['C', 'gL', 'VT', 'EL'],  # single allowed, multiple (same params for y) if type binned_lineplot
    #     "x_para": [
    #         "gL"
    #     ],  # single allowed, multiple (same params for y) if type binned_lineplot
    #     "x_ave": False,  # Weighted average over NGs. Works only for kind = regplot
    #     "y_ana": [
    #         "Coherence",
    #         "Granger Causality",
    #         # "GC as TE",
    #         "Transfer Entropy",
    #         # 'Euclidean Distance', # Note: Euclidean Distance cannot have accuracy
    #         "Simulation Error",
    #     ],  # multiple allowed => subplot columns, unless ave
    #     "y_mid": [
    #         "Comrad",
    #         "Bacon",
    #         # 'HiFi',
    #     ],  # single allowed, multiple (same mids for x) if type binned_lineplot
    #     # "y_para": ['C', 'gL', 'VT', 'EL'],  # single allowed, multiple (same params for x) if type binned_lineplot 'C','gL','EL','VT'
    #     "y_para": [
    #         "gL"
    #     ],  # single allowed, multiple (same params for x) if type binned_lineplot 'C','gL','EL','VT'
    #     "y_ave": False,  # Weighted average over NGs. Works only for kind = regplot
    #     "kind": "binned_lineplot",  # binned_lineplot, regplot
    #     "n_bins": 10,  # ignored for regplot
    #     "hue": "Midpoint",  # Midpoint or Parameter. If Midpoint is selected, each line is one midpoint and parameters will be combined. And vice versa. Ignored for regplot
    #     "compiled_results": True,  # x and y data from folder XX'_compiled_results'
    #     "compiled_type": "accuracy",  # mean, accuracy; falls back to mean if accuracy not found
    #     "draw_regression": False,  # only for regplot
    #     "order": 1,  # Regression polynomial fit order, only for regplot
    #     "draw_diagonal": False,  # only for regplot
    #     "xlog": False,
    #     "ylog": False,
    #     "sharey": True,
    # }

    # PM.viz.show_xy_plot(xy_plot_dict)

    """
    Show input-to-output classification confusion matrix
    midpoints Comrad, Bacon, HiFi; parameter 'C', 'gL', 'VT', 'EL', 'delay'
    """
    # PM.viz.show_IxO_conf_mtx(midpoint='Comrad', parameter='VT', ana_list=['Coherence', 'GrCaus', 'TransferEntropy', 'NormError'],
    #     ana_suffix_list=['sum', 'Information', 'TransfEntropy', 'SimErr'], par_value_string_list=['-44.0', '-46.0'],
    #     best_is_list=['max', 'max', 'max', 'min'])

    ##################################################################################
    ###### Analyze & Viz, array runs, multiple iterations, multiple paths ############
    ##################################################################################

    """
    Categorical plot of parametric data.
    Definitions for parametric plotting of multiple conditions/categories.
    First, define what data is going to be visualized in to_mpa_dict.
    Second, define how it is visualized in param_plot_dict.

    Limitations: 
        You cannot have analyzes as title AND inner_sub = True.
        For violinplot and inner_sub = True, N bin edges MUST be two (split view)

    outer : panel (distinct subplots) # analyzes, midpoints, parameters, controls
    inner : inside one axis (subplot) # midpoints, parameters, controls
    inner_sub : bool, further subdivision by value, such as mean firing rate
    inner_sub_ana : name of ana. This MUST be included into to_mpa_dict "analyzes"
    plot_type : parametric plot type # box

    compiled_results : data at compiled_results folder, mean over iterations

    inner_paths : bool (only inner available for setting paths). Provide comparison from arbitrary paths, e.g. controls. The 'inner' is ignored.
    inner_path_names: list of names of paths to compare.
    paths : provide list of tuples of full path parts to data folder. 
    E.g. [(path, 'Single_narrow_iterations_control', 'Bacon_gL_compiled_results'),] 
    The number of paths MUST be the same as the number of corresponding inner variables. 
    save_description: bool, if True, saves pd.describe() to csv files for each title into path/Description/
    """

    # param_plot_dict = {
    #     "title": "parameters",  # multiple allowed => each in separate figure
    #     "outer": "analyzes",  # multiple allowed => plt subplot panels
    #     "inner": "midpoints",  # multiple allowed => direct comparison
    #     "inner_sub": False,  # A singular analysis => subdivisions
    #     "inner_sub_ana": "Excitatory Firing Rate",  #  The singular analysis
    #     "bin_edges": [[0.001, 150], [150, 300]],
    #     "plot_type": "box",  # "violin" (2), "box", "strip", "swarm", "boxen", "point", "bar"
    #     "compiled_results": True, # True, False
    #     "sharey": False,
    #     "palette": "Greys",
    #     "inner_paths": False,
    #     # "inner_path_names": ["Comrad", "Bacon", "Bacon_EI", "Bacon_ALL"],
    #     "paths": [
    #         # (Path(root_path), Path(project), 'Single_narrow_iterations', 'Comrad_gL_compiled_results'),
    #         # (Path(root_path), Path(project), 'Single_narrow_iterations_control_EI', 'Bacon_gL_compiled_results'),
    #         ],
    #     "inner_stat_test": False,
    #     "save_description": False,
    #     "save_name": "description_simulated",
    #     "display_optimal_values": True,
    #     "optimal_value_foldername": optimal_value_foldername,
    #     "optimal_description_name": optimal_description_name
    # }
    # PM.viz.show_catplot(param_plot_dict)

    """
    ### Housekeeping ###. Do not comment out.
    """
    plt.show()

    ####################################################################
    ##### Automated csv generation, simulation and analysis ############
    ##### Useful e.g. for multiple input files              ############
    ####################################################################

    """
    Configuration for iterations, i.e. generating anat and phys csv
    files from midpoint files, running simulations in parallel subprocesses
    and running analysis with parallel threads. 
    Run optimal analysis provides values for input vs itself with given
    delays (optimal, bas case scenario), and each input vs all other inputs 
    with given delays (nonoptimal, worst case scenario).
    """

    # What to do
    create_csvs = 0
    run_simulation = 0
    run_analysis = 0
    run_optimal_analysis = 0  # Always nonparallel

    # Two parallel analyzes exist (None for non-parallel):
    # 'full_IxO' : create .gz with all inputs vs all outputs, then csv:s.
    # Valid IxO analyzes include Coherence, TransferEntropy, GrCaus, NormError
    # 'matched_IO' : create csv:s from matched input to output pairs. All analyzes are valid.
    parallel_analysis = True
    analysis_type = "full_IxO"
    create_log_file = True

    optimal_value_delays = [
        0,
        1000,
        500,
    ]  # list, interpreted as [min, max, number of steps]. Always nonparallel.

    max_n_iter = 10  # Max N repetitions in the project
    # For single input keep n_iters = 1 and  iter_start_idx = 0
    n_iters = max_n_iter
    iter_start_idx = 0

    # N padded zeros in iterated folder names. If 2 => Bacon_C_00, if 1 => Bacon_C_0
    pad_zeros = 2

    # Logic for iteration index list
    # Activate the bottom row for manual selection of iters e.g. [0, 5, 9]
    if iter_start_idx == 0 and n_iters == 1:
        iter_idx_list = None
    else:
        iter_idx_list = list(range(iter_start_idx, iter_start_idx + n_iters))
    # iter_idx_list = []

    # input_fname_prefix = f"noise_220309"
    input_fname_prefix = f"noise_210916_SingleSpike"
    input_fname_ci_suffix = f"_ci.mat"
    # time_ids = {"Comrad": "211118", "Bacon": "211222", "HiFi": "211118"}
    time_ids = {"Comrad": "221122", "Bacon": "221122", "HiFi": "221122"}

    anat_update_dict = {
        "workspace_path": f"{path}",
        # "import_connections_from": f"../in/connections_learned_6by6_ci.gz",
        "import_connections_from": f"../in/connections_SingleSpike_ci.gz",
        "run_in_cluster": 0,  # SIC(!)
        "cluster_job_file_path": "../csc_puhti_sv.job",
    }

    in_folder_full = PM.context.input_folder

    # Project physiology search space
    phys_update_dict = {
        "current_injection": [
            ["base_ci_path", nan, f"r'{in_folder_full}'", "", "variable"],
            [
                "ci_filename",
                nan,
                f"'{input_fname_prefix}{input_fname_ci_suffix}'",
                "",
                "variable",
            ],
        ],
        "C": [
            ["L4_CI_BC", "C", "{ 30.0 | 270.0 | 10.0 }", " * pF", "key"],
            ["L4_CI_SS", "C", "{ 30.0 | 130.0 | 10.0 }", " * pF", "key"],
        ],
        "gL": [
            ["L4_CI_BC", "gL", "{ 1.0 | 28.0 | 1.0 }", " * nS", "key"],
            ["L4_CI_SS", "gL", "{ 0.5 | 15.0 | 1.0 }", " * nS", "key"],
        ],
        "VT": [
            ["L4_CI_BC", "VT", "{ -65.0 | -15.0 | 3.0 }", " * mV", "key"],
            ["L4_CI_SS", "VT", "{ -67.0 | -35.0 | 3.0 }", " * mV", "key"],
        ],
        "EL": [
            ["L4_CI_BC", "EL", "{ -85.0 | -35.0 | 5.0 }", " * mV", "key"],
            ["L4_CI_SS", "EL", "{ -85.0 | -20.0 | 5.0 }", " * mV", "key"],
        ],
        "delay": [["delay", "delay_CI_CI", "{ 0.5 | 25 | 0.25 }", " * ms", "key"]],
    }

    # # Project physiology search space: classification_1D
    # phys_update_dict = {
    #     "current_injection": [
    #         ["base_ci_path", nan, f"r'{in_folder_full}'", "", "variable"],
    #         [
    #             "ci_filename",
    #             nan,
    #             f"'{input_fname_prefix}{input_fname_ci_suffix}'",
    #             "",
    #             "variable",
    #         ],
    #     ],
    #     "C": [["L4_CI_SS", "C", "{ 1.0 | 750.0 | 3.0 }", " * pF", "key"]],
    #     "gL": [
    #         ["L4_CI_SS", "C", "{ 10 & 125 }", " * pF", "key"],
    #         ["L4_CI_SS", "gL", "{ 0.0001 | 101.0 | 1.0 }", " * nS", "key"],
    #     ],
    #     "VT": [["L4_CI_SS", "VT", "{ -85.0 | -5.0 | 1.0 }", " * mV", "key"]],
    #     "EL": [
    #         ["L4_CI_SS", "C", "{ 10 & 125 }", " * pF", "key"],
    #         ["L4_CI_SS", "EL", "{ -150.0 | 150.0 | 3.0 }", " * mV", "key"],
    #     ],
    # }

    # # Dummy dict for dev and debugging
    # phys_update_dict = {
    #     "current_injection": [
    #         ["base_ci_path", nan, f"r'{in_folder_full}'", "", "variable"],
    #         [
    #             "ci_filename",
    #             nan,
    #             f"'{input_fname_prefix}{input_fname_ci_suffix}'",
    #             "",
    #             "variable",
    #         ],
    #     ],
    #     "C": [
    #         ["L4_CI_BC", "C", "{ 30.0 & 50.0  }", " * pF", "key"],
    #         ["L4_CI_SS", "C", "{ 30.0 & 50.0  }", " * pF", "key"],
    #     ],
    #     "gL": [
    #         ["L4_CI_BC", "gL", "{ 1.0 | 3.0 | 1.0 }", " * nS", "key"],
    #         ["L4_CI_SS", "gL", "{ 0.5 | 2.5 | 1.0 }", " * nS", "key"],
    #     ],
    # }

    PM.build_iterator(
        create_csvs=create_csvs,
        run_simulation=run_simulation,
        run_analysis=run_analysis,
        run_optimal_analysis=run_optimal_analysis,
        optimal_value_delays=optimal_value_delays,
        optimal_value_foldername=optimal_value_foldername,
        parallel_analysis=parallel_analysis,
        analysis_type=analysis_type,
        create_log_file=create_log_file,
        iter_idx_list=iter_idx_list,
        pad_zeros=pad_zeros,
        input_fname_prefix=input_fname_prefix,
        input_fname_ci_suffix=input_fname_ci_suffix,
        time_ids=time_ids,
        anat_update_dict=anat_update_dict,
        phys_update_dict=phys_update_dict,
    )

    PM.run_iterator()

    """
    Combine array analyzes data over multiple iterations. Generated by PM.build_iterator and
    PM.run_iterator methods
    # """
    # PM.ana.compile_analyzes_over_iterations("mean")  # 'mean', 'IxO_accuracy'

    if profile is True:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats(20)
