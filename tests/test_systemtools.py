# Builtin
import os
import sys
import shutil
from copy import deepcopy

# Testing
import pytest

from pathlib import Path

# Analysis
import pandas as pd
import numpy as np

# Dev
import pdb
import matplotlib.pyplot as plt

# Current package
from project.project_manager_module import ProjectManager
from data_io.data_io_module import DataIO
from construction.connection_translation_module import ConnectionTranslator
from context.context_module import Context
from analysis.analysis_module import Analysis
from analysis.statistics_module import Classifierbinomial
from analysis.cx_parser_module import CXParser


"""
To use this test you need to pip install -U pytest. 
Note that the -U will upgrade necessary dependencies for pytest.

Run pytest at SystemTools root, such as git repo root.

Simo Vanni 2021
"""

repo_path = Path.cwd()  # assumes cwd = SystemTools root
test_path = repo_path.joinpath("tests/out")

data_0_folder_name = Path("Comrad_C_0")
meta_fname = "metadata__20211008_1039431.gz"
csv_file_name = "MeanFR__20211008_1039431.csv"
mat_file_name = "noise_210916_SingleSpike_0.mat"
matlab_workspace_file = "workspace.mat"
conn_skeleton_file_in = "connections_SingleSpike_ci.gz"
single_data_file_name = "Comrad_C_0_results_20211008_1039431_C300pF_C300pF.gz"
single_gz_file_path = repo_path.joinpath(
    test_path, data_0_folder_name, single_data_file_name
)
data_folder_path = repo_path.joinpath(test_path, data_0_folder_name)

input_folder = "../in"

substring_results = "results"
substring_exclude = "400pF.gz"
substring_in = "connections"
substring_test_root = "tests"

midpoint = "Comrad"
parameter = "C_0"
output_folder = Path(midpoint + "_" + parameter)
conn_file_out = "connections_SingleSpike_ci.gz"

input_filename = "noise_210916_SingleSpike_0.mat"
NG_name = "NG3_L4_SS_L4"
t_idx_start = 2000
t_idx_end = -2000

to_mpa_dict = {
    "midpoints": ["Comrad"],
    "parameters": ["C"],
    "analyzes": ["Excitatory Firing Rate", "Inhibitory Firing Rate"],
}

pm_kw_dict = dict(
    path=test_path,
    input_folder=input_folder,
    output_folder=output_folder,
    matlab_workspace_file=matlab_workspace_file,
    conn_skeleton_file_in=conn_skeleton_file_in,
    conn_file_out=conn_file_out,
    input_filename=input_filename,
    midpoint=midpoint,
    parameter=parameter,
    NG_name=NG_name,
    t_idx_start=t_idx_start,
    t_idx_end=t_idx_end,
    to_mpa_dict=to_mpa_dict,
)
PM = ProjectManager(**pm_kw_dict)

to_mpa_dict_persist = deepcopy(to_mpa_dict)
# ###################
# # Unit tests
# ###################

# Context


def test_context():
    pm_kw_dict_copy = pm_kw_dict.copy()
    con = Context(pm_kw_dict)
    assert isinstance(con, Context)
    context = con.set_context(_properties_list=[])
    context.__dict__.pop("validated_properties")
    assert sorted(pm_kw_dict_copy.keys()) == sorted(context.__dict__.keys())


# DataIO.

IO = PM.data_io


def test_io_properties():
    assert isinstance(IO._properties_list, list)


def test_check_cadidate_file():
    checked_filepath = IO._check_cadidate_file(repo_path, single_gz_file_path)
    assert Path(checked_filepath) == single_gz_file_path


@pytest.mark.xfail(reason="not stabilized n files yet")
def test_listdir_loop():
    file_list = IO.listdir_loop(data_folder_path, data_type="", exclude=None)
    assert len(file_list) == 9
    file_list = IO.listdir_loop(
        data_folder_path, data_type=substring_results, exclude=None
    )
    assert len(file_list) == 2
    file_list = IO.listdir_loop(
        data_folder_path, data_type=substring_results, exclude=substring_exclude
    )
    assert len(file_list) == 1


def test_most_recent():
    csv_file_full_path = Path.joinpath(data_folder_path, csv_file_name)
    Path(csv_file_full_path).touch()
    file_list = IO.most_recent(data_folder_path)
    assert csv_file_name in str(file_list)


def test_parse_path():
    fullpath = IO.parse_path(single_data_file_name)
    assert Path(fullpath) == single_gz_file_path
    fullpath = IO.parse_path(None, data_type=substring_in)
    assert substring_in in fullpath.as_posix()
    fullpath = IO.parse_path(None, data_type=substring_test_root)
    assert substring_test_root in fullpath.as_posix().lower()


# @pytest.mark.skip(reason="slow due to warning")


def test_get_data():
    data = IO.get_data(filename=single_data_file_name)  # gz
    assert isinstance(data, dict)
    data = IO.get_data(filename=mat_file_name)  # mat
    assert isinstance(data, dict)


def test_get_csv_data_as_df():
    single_csv_file_path = repo_path.joinpath(test_path, data_0_folder_name)
    (
        data0_df,
        data_df_compiled,
        independent_var_col_list,
        dependent_var_col_list,
        time_stamp,
    ) = IO.get_csv_as_df(
        folder_name=None, csv_path=single_csv_file_path, include_only=None
    )
    assert isinstance(data0_df, pd.DataFrame)


def test_read_input_matfile():
    data = IO.read_input_matfile(filename=mat_file_name)
    assert isinstance(data, np.ndarray)


# ConnectionTranslator


def test_ct_construction():
    # Already built before the tests
    assert isinstance(PM.ct, ConnectionTranslator)
    assert isinstance(PM.ct.context, Context)
    assert isinstance(PM.ct.data_io, DataIO)


def test_deneve_replace_conn():
    orig_conn_file = "orig_connections_SingleSpike_ci.gz"
    orig_save_full = Path.joinpath(PM.context.input_folder, orig_conn_file)
    save_full = Path.joinpath(PM.context.input_folder, PM.context.conn_file_out)
    shutil.copy2(orig_save_full, save_full)
    mtime = os.path.getmtime(save_full)
    conn = PM.data_io.get_data(save_full)

    PM.ct.deneve_replace_conn()

    mtime2 = os.path.getmtime(save_full)
    assert mtime != mtime2

    conn2 = PM.data_io.get_data(save_full)
    assert all([key == key2 for (key, key2) in zip(conn, conn2)])


def test_scale_with_constant():
    constant_value = 1e-9
    connections_np = np.array([[1, 2, 3], [4, 5, 6]])
    connections_np_out = PM.ct.scale_with_constant(
        connections_np, constant_value=constant_value
    )
    assert np.sum(connections_np_out - (connections_np * constant_value)) == 0


def test_scale_values():
    source_data = np.array([[1, 2, 3], [4, 5, 6]])
    target_data = np.array([0, 2])  # default is [0, 1]
    scaled_data = PM.ct.scale_values(source_data, target_data=None)
    assert np.ptp(scaled_data) == np.ptp(target_data / 2)
    scaled_data = PM.ct.scale_values(source_data, target_data=target_data)
    assert np.ptp(scaled_data) == np.ptp(target_data)


def test_deneve_create_current_injection_input_filename():
    # Read existing Input
    input_filename = PM.context.input_filename
    assert input_filename is not None
    assert input_filename.suffix == ".mat"


def test_deneve_create_current_injection():
    input_filename = PM.context.input_filename

    input_filename_full = Path.joinpath(PM.context.input_folder, input_filename)
    current_injection_filename_full = Path(
        input_filename_full.as_posix()[:-4] + "_ci.mat"
    )
    orig_current_injection_filename_full = Path.joinpath(
        PM.context.input_folder, "orig_noise_210916_SingleSpike_0_ci.mat"
    )

    # shutil.copy2(orig_current_injection_filename_full, current_injection_filename_full)

    ci_dict = PM.data_io.get_data(orig_current_injection_filename_full)
    PM.ct.deneve_create_current_injection(randomize=False)
    ci_dict2 = PM.data_io.get_data(current_injection_filename_full)
    np.sum(
        np.abs(ci_dict["injected_current"]) - np.abs(ci_dict2["injected_current"])
    ) < 1e-10


# Analysis

ana = PM.ana

# Prep for signal analysis
data_dict = ana.data_io.get_data(single_gz_file_path)
diag_filename = "grcaus_FitDiag.txt"
ana.gc_dg_filename = Path.joinpath(ana.context.output_folder, diag_filename)
input_file_full_path = Path.joinpath(test_path, Path(input_folder), input_filename)
source_signal = ana.data_io.get_data(input_file_full_path)
source_signal_np = source_signal["stimulus"].T  # Note the shape (3, 20000) => .T
dt = ana.cxparser.get_dt(data_dict)
target_group = ana.context.NG_name
vm_unit = ana.cxparser.get_vm_by_interval(
    data_dict, target_group, t_idx_start=0, t_idx_end=None
)
target_signal_np = vm_unit / vm_unit.get_best_unit()
_source = source_signal_np[:, 0]
_target = target_signal_np[:, 0]
nsamples = ana.cxparser.get_n_samples(data_dict)
nperseg = nsamples // 6
samp_freq = 1.0 / dt
NG_list = [n for n in data_dict["spikes_all"].keys() if "NG" in n]


def test_analysis_construction():
    # Already built before the tests
    assert isinstance(ana, Analysis)
    assert isinstance(ana.context, Context)
    assert isinstance(ana.data_io, DataIO)


def test_scaler():
    source_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    data_scaled = ana.scaler(source_data, scale_type="standard", feature_range=[-1, 1])
    assert sum(data_scaled[1, :]) == 0
    data_scaled = ana.scaler(source_data, scale_type="minmax", feature_range=[0, 1])
    assert np.min(data_scaled) == 0 and np.max(data_scaled) == 1


def test__downsample():
    data = np.arange(0, 1, 0.1)
    data_ds = ana._downsample(data, downsampling_factor=2)
    assert len(data_ds) == len(data) / 2


def test__correlation_lags():
    in1_len = 5
    in2_len = 2
    lags = ana._correlation_lags(in1_len, in2_len, mode="full")
    assert np.sum(lags - np.array([-1, 0, 1, 2, 3, 4])) == 0


def test_get_coherence_of_two_signals():

    (
        f,
        Cxy,
        Pwelch_spec_x,
        Pwelch_spec_y,
        Pxy,
        lags,
        corr,
        coherence_sum,
        _source_scaled,
        _target_scaled,
    ) = ana.get_coherence_of_two_signals(
        _source, _target, samp_freq=samp_freq, nperseg=nperseg, high_cutoff=100
    )

    assert (
        len(f) == len(Cxy) == len(Pwelch_spec_x) == len(Pwelch_spec_y) == len(Pxy) == 34
    )
    assert len(lags) == len(corr) == 39999
    assert np.floor(coherence_sum) == 16


def test_end2idx():
    t_idx_end = -2000
    n_samples = 20000
    t_idx_end_abs = ana.end2idx(t_idx_end, n_samples)
    assert t_idx_end_abs == n_samples + t_idx_end


def test_pivot_to_2D_dataframe():

    csv_file_full_path = Path.joinpath(data_folder_path, csv_file_name)
    data_df = pd.read_csv(csv_file_full_path)
    value_column_name = "MeanFR_NG1_L4_CI_SS_L4"
    # Get 2 dims for viz
    index_column_name = "Dimension-1 Value"
    column_column_name = "Dimension-2 Value"

    df_2d = ana.pivot_to_2d_dataframe(
        data_df,
        index_column_name=index_column_name,
        column_column_name=column_column_name,
        value_column_name=value_column_name,
    )

    assert np.all(df_2d.columns == [30, 40])
    assert np.all(df_2d.index == [30.0, 40.0])


def test_get_extremes():
    csv_file_full_path = Path.joinpath(data_folder_path, csv_file_name)
    data_df = pd.read_csv(csv_file_full_path)
    value_column_name = "MeanFR_NG1_L4_CI_SS_L4"
    (
        data_nd_array,
        x_label,
        y_label,
        df_2d,
        x_values,
        y_values,
        min_value,
        min_idx,
        max_value,
        max_idx,
    ) = ana.get_df_extremes(data_df, value_column_name, two_dim=True)

    assert data_nd_array.size == 4
    assert x_label == "C_1"
    assert y_label == "C"
    assert np.floor(min_value) == 273
    assert np.floor(max_value) == 295


def test_get_cross_corr_latency():

    (
        f,
        Cxy,
        Pwelch_spec_x,
        Pwelch_spec_y,
        Pxy,
        lags,
        corr,
        coherence_sum,
        _source_scaled,
        _target_scaled,
    ) = ana.get_coherence_of_two_signals(
        _source, _target, samp_freq=samp_freq, nperseg=nperseg, high_cutoff=100
    )

    shift_in_seconds = ana.get_cross_corr_latency(lags, corr, dt)

    assert shift_in_seconds < 0.03


def test__init_grcaus_regression_QA():

    ana.gc_dg_filename.unlink(missing_ok=True)  # dev

    ana.extra_ana_args["GrCaus_args"]["save_gc_fit_dg_and_QA"] = False
    regression_error_data_list = ana._init_grcaus_regression_QA()
    # If file found but the test_path has changed, replace with default
    if str(test_path) not in regression_error_data_list[0]:
        if single_gz_file_path.stem in regression_error_data_list[0]:
            regression_error_data_list = [str(single_gz_file_path)]
    assert hasattr(ana, "_tmp_regression_error_full_path")
    assert Path(regression_error_data_list[0]) == single_gz_file_path
    assert not ana.gc_dg_filename.is_file()
    ana.extra_ana_args["GrCaus_args"]["save_gc_fit_dg_and_QA"] = True
    regression_error_data_list = ana._init_grcaus_regression_QA()
    assert ana.gc_dg_filename.is_file()
    # Cleanup
    ana.gc_dg_filename.unlink()


def test__get_passing():

    value = 1
    threshold = 2
    passing = ana._get_passing(value, threshold, passing_goes="over")
    assert passing == "FAIL"
    passing = ana._get_passing(value, threshold, passing_goes="under")
    assert passing == "PASS"
    threshold = 0.5
    passing = ana._get_passing(value, threshold, passing_goes="over")
    assert passing == "PASS"
    passing = ana._get_passing(value, threshold, passing_goes="under")
    assert passing == "FAIL"
    threshold = [0.5, 2]
    passing = ana._get_passing(value, threshold, passing_goes="both")
    assert passing == "PASS"
    value = 3
    passing = ana._get_passing(value, threshold, passing_goes="both")
    assert passing == "FAIL"


def test__preprocess_for_info_analyzes():

    (source_signal_pp, target_signal_pp,) = ana._preprocess_for_info_analyzes(
        source_signal_np,
        target_signal_np,
        ana.extra_ana_args["GrCaus_args"]["downsampling_factor"],
        nsamples,
        t_idx_start=ana.context.t_idx_start,
        t_idx_end=ana.context.t_idx_end,
    )

    assert source_signal_pp.shape == target_signal_pp.shape == (400, 3)


def test__analyze_grcaus():

    ana.extra_ana_args["GrCaus_args"]["save_gc_fit_dg_and_QA"] = False

    (
        GrCaus_information,
        GrCaus_p,
        GrCaus_latency,
        target_signal_entropy,
        MeanGrCaus_fitQA,
        GrCaus_InfoAsTE,
    ) = ana._analyze_grcaus(
        data_dict,
        source_signal_np,
        dt,
        target_group,
        verbose=False,
        return_only_infomatrix=False,
        known_regression_failure=False,
    )

    assert np.floor(GrCaus_information) == 4
    assert GrCaus_latency < 0.08
    assert ana.round_to_n_significant(target_signal_entropy, 2) == 1.3
    assert ana.round_to_n_significant(MeanGrCaus_fitQA, 2) == 4.7

    ana.gc_dg_filename.unlink(missing_ok=True)  # dev


def test__pin_transfer_entropy():

    pin_te = ana._pin_transfer_entropy(_target, _source, 1, n_states=2)
    assert 0.0002 < pin_te < 0.0003


def test__vm_entropy():

    entropy = ana._vm_entropy(target_signal_np, n_states=None, bins=None)
    assert 0.8 < entropy < 0.9


def test__get_MSE_1d():

    # CxSystem2
    Error = ana._get_MSE_1d(
        Input=_source,
        simulated_output_vm=_target,
        results_filename=None,
        data_dict=data_dict,
        simulation_engine="CxSystem",
        readout_group="NG1",
        decoding_method="least_squares",
        output_type="estimated",
    )
    assert 0.7 < Error < 0.8


def test_get_MSE():

    # CxSystem2
    Error, foo, foo1 = ana.get_MSE(
        Input=None,  # Reads input matfile
        results_filename=single_data_file_name,
        data_dict=None,
        simulation_engine="CxSystem",
        readout_group="NG1",
        decoding_method="least_squares",
        output_type="simulated",
    )

    assert 1 < Error < 1.1


def test__analyze_meanfr():

    mean_fr = ana._analyze_meanfr(data_dict, NG_list[1])
    assert 291 < mean_fr < 292


def test__analyze_euclid_dist():

    mean_dist = ana._analyze_euclid_dist(data_dict, source_signal_np, NG_name)
    assert 0.4 < mean_dist < 0.5


def test__analyze_meanvm():

    mean_vm = ana._analyze_meanvm(data_dict, NG_list[3])
    assert -68 < mean_vm / mean_vm.get_best_unit() < -67


def test__analyze_transfer_entropy():
    te, te_latency = ana._analyze_transfer_entropy(
        data_dict, source_signal_np, dt, target_group
    )

    assert 0.18 < te < 0.19
    assert 0.013 < te_latency < 0.014


def test__analyze_coherence():
    coh_sum, coh_lat = ana._analyze_coherence(
        data_dict, source_signal_np, dt, target_group
    )
    assert 20 < coh_sum < 21
    assert 0.005 < coh_lat < 0.006


def test__analyze_normerror():
    EstimErr_E, EstimErr_I, SimErr_O = ana._analyze_normerror(
        data_dict, source_signal_np
    )

    assert 1 < SimErr_O < 1.1


def test__analyze_classification_performance():
    y_data = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    acc = ana._analyze_classification_performance(y_data, y_true=None)
    assert acc == 1.0
    y_true = y_data / 10
    y_data = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
    acc = ana._analyze_classification_performance(y_data, y_true=y_true)
    assert acc == 1.0 / 3


# Get metadata and output filenames
meta_fname_full = ana.data_io.parse_path(
    meta_fname, data_type="metadata", exclude="cluster_metadata"
)
# meta_root = Path.joinpath(meta_fname_full.parent, meta_fname_full.stem)
# meta_ext = meta_fname_full.suffix
meta_fname_updated, data_df = ana.update_metadata(meta_fname_full)
meta_fname_updated = Path(meta_fname_updated)
metadataroot = Path.joinpath(meta_fname_full.parent, meta_fname_full.stem)


def test__get_analyzed_array_as_df():

    ana_data_df = ana._get_analyzed_array_as_df(data_df, analysisHR="MeanFR")
    assert ana_data_df.shape == (4, 9)
    describe_df = ana_data_df.describe()
    assert 284 < describe_df.loc["mean"]["MeanFR_" + NG_list[1]] < 285


# @pytest.mark.skip(reason="slow")
def test__get_analyzed_full_array_as_np():
    ana_list = ["Coherence", "TransferEntropy", "GrCaus", "NormError"]
    ana_data_dict = ana._get_analyzed_full_array_as_np(data_df, analyzes_list=ana_list)

    assert ana_data_dict["data_dims"] == (3, 3, 4)
    assert 344 < np.sum(ana_data_dict["Coherence_NG3_L4_SS_L4_Sum"]) < 345
    assert (
        2.5
        < np.nansum(ana_data_dict["TransferEntropy_NG3_L4_SS_L4_TransfEntropy"])
        < 2.6
    )
    assert 56 < np.sum(ana_data_dict["NormError_SimErr"]) < 57
    assert 56 < np.sum(ana_data_dict["GrCaus_NG3_L4_SS_L4_Information"]) < 57


def test__get_midpoint_parameter_dict():

    results_folder_suffix = "_compiled_results"
    midpoint_parameter_dict = ana._get_midpoint_parameter_dict(
        results_folder_suffix=results_folder_suffix
    )
    test_key = to_mpa_dict["midpoints"][0] + "_" + to_mpa_dict["parameters"][0]

    assert test_key in midpoint_parameter_dict.keys()
    assert len(midpoint_parameter_dict[test_key]) == 2


def test_get_PCA():

    row_selection = [1]
    folder_name = None
    (
        data0_df,
        data_df_compiled,
        independent_var_col_list,
        dependent_var_col_list,
        time_stamp,
    ) = ana.data_io.get_csv_as_df(folder_name=folder_name)

    # Combine dfs
    df_for_barplot, col_list, min_values, max_values = ana._get_system_profile_metrics(
        data_df_compiled, independent_var_col_list
    )

    # Normalize magnitudes
    values_np = df_for_barplot[col_list].values  # returns a numpy array
    values_np_scaled = ana.scaler(values_np, scale_type="minmax", feature_range=[0, 1])
    df_for_barplot[col_list] = values_np_scaled

    # extra points are the original dimensions, to be visualized
    extra_points = np.vstack([np.eye(len(col_list)), -1 * np.eye(len(col_list))])

    # Get PCA of data. Note option for extra_points=extra_points_df
    values_pca, foo1, foo2, foo3 = ana.get_PCA(
        values_np, col_names=col_list, extra_points=extra_points
    )

    assert -2.7 < np.min(values_pca) < -2.6
    assert 2.8 < np.max(values_pca) < 2.9


def test_analyze_arrayrun():

    analysis = "Coherence"
    [coh_fname] = ana.data_io.listdir_loop(
        data_folder_path, data_type=analysis, exclude="IxO_analysis"
    )
    mtime = os.path.getmtime(coh_fname)
    ana.analyze_arrayrun(metadata_filename=meta_fname_full, analysis=analysis)
    mtime2 = os.path.getmtime(coh_fname)
    assert mtime < mtime2


def test_analyze_IxO_array():

    analysis_filename_prefix = "IxO_analysis_"
    _filename = str(metadataroot).replace("metadata_", analysis_filename_prefix)
    IxO_dict_filename_out = Path(_filename + ".gz")

    if IxO_dict_filename_out.is_file():
        Path(IxO_dict_filename_out).unlink()

    ana_list = ["Coherence"]
    ana.analyze_IxO_array(metadata_filename=meta_fname_full, analyzes_list=ana_list)
    assert IxO_dict_filename_out.is_file()


def analyze_TE_drift():
    analysis_filename_prefix = "TEDrift_analysis_TEDrift_"
    _filename = str(metadataroot).replace("metadata_", analysis_filename_prefix)
    _csvname = str(metadataroot).replace("metadata_", "TEDrift_")
    TEDrift_dict_filename_out = Path(_filename + ".gz")
    TEDrift_csv_filename_out = Path(_csvname + ".csv")

    ana.analyze_TE_drift(metadata_filename=meta_fname_full)
    assert TEDrift_dict_filename_out.is_file()
    assert TEDrift_csv_filename_out.is_file()
    Path(TEDrift_dict_filename_out).unlink()
    Path(TEDrift_csv_filename_out).unlink()


@pytest.mark.skip(reason="method inactive")
def test_collate_best_values():
    collated_best_values_df, x_val, y_val = ana.collate_best_values(
        midpoint=None,
        parameter=None,
        analyzes=None,
        save_as_csv=False,
        output_folder=None,
    )
    assert collated_best_values_df["best_is"].values.tolist() == [
        "min",
        "max",
        "max",
        "max",
    ]


def test_compile_analyzes_over_iterations():

    _filename = "MeanFR_mean.csv"
    _folder = "Comrad_C_compiled_results"
    mean_file_full = Path.joinpath(test_path, _folder, _filename)

    mtime = os.path.getmtime(mean_file_full)
    ana.compile_analyzes_over_iterations()
    mtime2 = os.path.getmtime(mean_file_full)
    assert mtime < mtime2

    df = pd.read_csv(mean_file_full)
    df_val = df[["MeanFR_NG1_L4_CI_SS_L4_mean", "MeanFR_NG2_L4_CI_BC_L4_mean"]]

    assert 271 < np.min(df_val.values) < 272
    assert 401 < np.max(df_val.values) < 402


# statistics_module
def test_statistics():
    y_data = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    true_ixo = np.eye(3, 3)
    classtest = Classifierbinomial(y_data, key_matrix=true_ixo)
    p_value = classtest.binomial_probability()
    alpha_confidence = 0.05
    (
        accuracy,
        lower_bound,
        upper_bound,
    ) = classtest.accuracy_confidence_interval(alpha_confidence)

    assert p_value == 0
    assert accuracy == 1.0
    assert lower_bound == 0.884
    assert upper_bound == 1.0


# CxParser

cxparser = CXParser()


def test_get_dt():
    dt = cxparser.get_dt(data_dict)
    assert dt == 0.0001


def test_get_nsamples():
    n_samples = cxparser.get_n_samples(data_dict)
    assert n_samples == 20000


def test_get_namespace_variable():
    val = cxparser.get_namespace_variable(data_dict, NG_name, variable_name="taum_soma")
    assert 31 < val / val.get_best_unit() < 32


def test_get_vm_by_interval():
    vm = cxparser.get_vm_by_interval(
        data_dict, NG=NG_name, t_idx_start=0, t_idx_end=None
    )

    assert vm.shape == (20000, 3)
    assert -84 < np.min(vm / vm.get_best_unit()) < -83
    assert -56 < np.max(vm / vm.get_best_unit()) < -55


# Currents not tested, because there are no currents monitored in the test data
def test__get_spikes_by_interval():

    N_neurons, spikes, dt = cxparser._get_spikes_by_interval(
        data_dict, NG=NG_list[1], t_idx_start=0, t_idx_end=20000
    )
    assert N_neurons == 300
    assert len(spikes) == 173184
    assert dt == 0.0001

    N_neurons, spikes, dt = cxparser._get_spikes_by_interval(
        data_dict, NG=NG_list[1], t_idx_start=4000, t_idx_end=16000
    )

    assert len(spikes) == 105096


# Viz

"""
Every data for visualization is evaluated by data_is_valid method. The figures are evaluated manually by commenting out plt.close('all') after if __name__ == "__main__": For automatic testing, we assert that figure objects are generated.
"""
viz = PM.viz
plt.close("all")


def test_data_is_valid():

    assert viz.data_is_valid(np.array([1, 2, 3]))
    assert not viz.data_is_valid(np.array([]))
    assert not viz.data_is_valid("wrong kind of data")


def test_show_readout_on_input():

    num_figures = len(plt.get_fignums())
    viz.show_readout_on_input(
        results_filename=single_gz_file_path,
        normalize=False,
        unit_idx_list=[0],
        savefigname="",
    )
    num_figures2 = len(plt.get_fignums())

    assert num_figures < num_figures2


def test_show_spikes():

    num_figures = len(plt.get_fignums())
    viz.show_spikes(results_filename=single_gz_file_path, savefigname="")
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2


def test_show_analog_results():

    num_figures = len(plt.get_fignums())
    viz.show_analog_results(
        results_filename=single_gz_file_path,
        savefigname="",
        param_name="vm",
        startswith="NG3",
        neuron_index=None,
    )
    num_figures2 = len(plt.get_fignums())

    assert num_figures < num_figures2


# No currents in test data, skipped show_currents.


def test_show_conn():

    num_figures = len(plt.get_fignums())
    viz.show_conn(conn_file=conn_file_out, hist_from=None, savefigname="")
    num_figures2 = len(plt.get_fignums())

    assert num_figures < num_figures2


def test_show_analyzed_arrayrun():

    num_figures = len(plt.get_fignums())
    viz.show_analyzed_arrayrun(
        csv_filename=csv_file_name, analysis="MeanFR", variable_unit="Hz", NG_id_list=[]
    )
    num_figures2 = len(plt.get_fignums())

    assert num_figures < num_figures2


def test_show_input_to_readout_coherence():

    num_figures = len(plt.get_fignums())
    viz.show_input_to_readout_coherence(
        results_filename=None, savefigname="", signal_pair=[0, 0]
    )
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2


def test_show_estimate_on_input():

    num_figures = len(plt.get_fignums())
    viz.show_estimate_on_input(
        results_filename=None,
        simulation_engine="cxsystem",
        readout_group="E",
        decoding_method="least_squares",
        output_type="estimated",
        unit_idx_list=[0],
    )
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2


def test_system_polar_bar():

    num_figures = len(plt.get_fignums())
    viz.system_polar_bar(row_selection=1, folder_name=None)
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2


def test_show_catplot():

    viz.context.to_mpa_dict = deepcopy(to_mpa_dict_persist)
    param_plot_dict = {
        "title": "parameters",
        "outer": "analyzes",
        "inner": "midpoints",
        "inner_sub": False,
        "inner_sub_ana": "Excitatory Firing Rate",
        "bin_edges": [[270, 280], [280, 290], [290, 300]],
        "plot_type": "box",
        "compiled_results": True,
        "sharey": False,
        "palette": "Greys",
        "inner_paths": False,
        "paths": [],
        "save_description": False,
        "display_optimal_values": False,
        "inner_stat_test": False,
    }

    num_figures = len(plt.get_fignums())
    viz.show_catplot(param_plot_dict)
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2

    param_plot_dict["inner_sub"] = True

    num_figures = len(plt.get_fignums())
    viz.show_catplot(param_plot_dict)
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2


def test_show_xy_plot():

    viz.context.to_mpa_dict = deepcopy(to_mpa_dict_persist)
    xy_plot_dict = {
        "x_ana": [
            "Excitatory Firing Rate"
        ],  # multiple allowed => subplot rows, unless ave
        "x_mid": "Comrad",  # multiple allowed
        "x_para": "C",  # multiple allowed
        "x_ave": False,  # All x_ana data will be averaged
        "y_ana": [
            "Inhibitory Firing Rate"
        ],  # multiple allowed => subplot columns, unless ave
        "y_mid": "Comrad",  # multiple allowed
        "y_para": "C",  # multiple allowed
        "y_ave": False,  # All y_ana data will be averaged
        "compiled_results": True,  # both x and y data
        "compiled_type": "mean",  # mean, accuracy
        "xlog": False,
        "ylog": False,
        "draw_regression": True,
        "sharey": False,
        "order": 1,  # Regression polynomial fit order
        "kind": "regplot",
        "draw_diagonal": False,
    }

    num_figures = len(plt.get_fignums())
    viz.show_xy_plot(xy_plot_dict)
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2

    xy_plot_dict["x_ave"] = True

    num_figures = len(plt.get_fignums())
    viz.show_xy_plot(xy_plot_dict)
    num_figures2 = len(plt.get_fignums())
    assert num_figures < num_figures2


# Teardown code
if "_updated" in meta_fname_updated.stem:
    Path(meta_fname_updated).unlink()
plt.close("all")


if __name__ == "__main__":
    test__init_grcaus_regression_QA()

    plt.show()

