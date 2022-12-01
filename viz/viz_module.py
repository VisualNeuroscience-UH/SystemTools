# Analysis
from cProfile import label
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import colour

# This computer git repos
# from system_analysis import Analysis
from viz.viz_base_module import VizBase

# Builtin
from pathlib import Path
from functools import reduce
import os
import copy
from itertools import chain
import time
import pdb

"""
Module for visualization

Developed by Simo Vanni 2020-2022
"""


class Viz(VizBase):

    cmap = "gist_earth"  # viridis or cividis would be best for color-blind
    _properties_list = [
        "path",
        "output_folder",
        "input_filename",
        "midpoint",
        "parameter",
        "NG_name",
        "t_idx_start",
        "t_idx_end",
        "conn_file_out",
        "to_mpa_dict",
    ]

    save_figure_with_arrayidentifier = None

    def __init__(self, context, data_io, ana, cxparser, **kwargs) -> None:

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        self._ana = ana
        self._cxparser = cxparser

        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def ana(self):
        return self._ana

    @property
    def cxparser(self):
        return self._cxparser

    def data_is_valid(self, data, accept_empty=False):

        try:
            data = data / data.get_best_unit()
        except:
            pass

        if accept_empty == True:
            is_valid = isinstance(data, np.ndarray)
        else:
            is_valid = isinstance(data, np.ndarray) and data.size > 0

        return is_valid

    # Helper functions
    def _build_columns(self, data_dict, new_data, Ntimepoints, datatype):

        tmp_dict = copy.deepcopy(data_dict)
        # columns for input
        new_data_shape = np.asarray(new_data.shape)
        new_data_time_dim = np.where(new_data_shape == Ntimepoints)
        if new_data_time_dim[0] == 1:
            new_data = new_data.T
            new_data_shape = np.asarray(new_data.shape)

        for ch_idx in np.arange(new_data_shape[1]):
            ch_name = f"{datatype}_ch_{ch_idx}"
            ch_data = new_data[:, ch_idx]
            tmp_dict[ch_name] = ch_data

        dims = new_data_shape[1]
        return tmp_dict, dims

    def _string_on_plot(
        self, ax, variable_name=None, variable_value=None, variable_unit=None
    ):

        plot_str = f"{variable_name} = {variable_value:6.2f} {variable_unit}"
        ax.text(
            0.05,
            0.95,
            plot_str,
            fontsize=8,
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="Square,pad=0.2", fc="white", ec="white", lw=1),
        )

    def _get_cut_data(self, df, start_idx, end_idx, nsamples):
        drop_list_start = np.arange(0, start_idx).tolist()
        drop_list_end = np.arange(end_idx, nsamples).tolist()
        drop_list = drop_list_start + drop_list_end
        df = df.drop(drop_list)
        return df

    def _get_n_neurons_and_data_array(
        self, data, this_group, param_name, neuron_index=None
    ):

        assert neuron_index is None or (
            isinstance(neuron_index, dict)
            and (
                isinstance(neuron_index[this_group], int)
                or isinstance(neuron_index[this_group], list)
            )
        ), """ neuron index for each group must be either None, dict[group_name]=int  
        eg {"NG1_L4_CI_SS_L4" : 150} or dict[group_name]=list of ints"""

        if neuron_index is None:
            N_monitored_neurons = data[f"{param_name}_all"][this_group][
                f"{param_name}"
            ].shape[1]
            this_data = data[f"{param_name}_all"][this_group][f"{param_name}"]
        elif isinstance(neuron_index[this_group], int):
            N_monitored_neurons = 1
            this_data = data[f"{param_name}_all"][this_group][f"{param_name}"][
                :, neuron_index[this_group]
            ]
        elif isinstance(neuron_index[this_group], list):
            N_monitored_neurons = len(neuron_index[this_group])
            this_data = data[f"{param_name}_all"][this_group][f"{param_name}"][
                :, neuron_index[this_group]
            ]

        return N_monitored_neurons, this_data

    def _pc2rgb(self, cardinal_points, additional_points):
        # Get transformation matrix between 2D cardinal points as corners of a rectangular space
        # [left lower, right upper, left upper, right lower]. Map this space to RGB gamut in CIExy color space.
        # Transform additional points between the spaces
        # returns the additional points

        # Major color axis contained in RGB gamut in CIE xy coordinates
        CIE_cardinal_points_list = [[0.6, 0.35], [0.3, 0.5], [0.2, 0.15], [0.45, 0.45]]
        CIE_cardinal_points_df = pd.DataFrame(
            CIE_cardinal_points_list,
            index=["red", "green", "blue", "yellow"],
            columns=["CIE_x", "CIE_y"],
        )
        CIE_matrix = CIE_cardinal_points_df.values

        # Get transformation matrix X between PC dimensions and CIExy coordinates
        PC_matrix_ones = np.c_[cardinal_points, np.ones([4, 1])]

        CIE_matrix_ones = np.c_[CIE_matrix, np.ones([4, 1])]
        transf_matrix = np.linalg.lstsq(PC_matrix_ones, CIE_matrix_ones)[0]

        CIE_space = (
            np.c_[additional_points, np.ones(additional_points.shape[0])]
            @ transf_matrix
        )
        CIEonXYZ = colour.xy_to_XYZ(CIE_space[:, :2])
        CIEonRGB = colour.XYZ_to_sRGB(CIEonXYZ / 100)

        # Get standard scaler
        CIE_standard_space = PC_matrix_ones @ transf_matrix
        CIEonXYZ_standard_space = colour.xy_to_XYZ(CIE_standard_space[:, :2])
        CIEonRGB_standard_space = colour.XYZ_to_sRGB(CIEonXYZ_standard_space / 100)
        standard_min = np.min(CIEonRGB_standard_space)
        standard_ptp = np.ptp(CIEonRGB_standard_space)

        # Scale brighter
        # RGB_points = (CIEonRGB - np.min(CIEonRGB)) / np.ptp(CIEonRGB)
        RGB_points_scaled = (CIEonRGB - standard_min) / standard_ptp
        # clip to [0, 1]
        RGB_points = np.clip(RGB_points_scaled, 0, 1)

        return RGB_points

    def figsave(self, figurename="", myformat="png", subfolderpath="", suffix=""):

        # Saves current figure to working dir
        plt.rcParams["svg.fonttype"] = "none"  # Fonts as fonts and not as paths
        plt.rcParams["ps.fonttype"] = "type3"  # Fonts as fonts and not as paths

        # Confirm pathlib type
        figurename = Path(figurename)
        subfolderpath = Path(subfolderpath)

        if myformat[0] == ".":
            myformat = myformat[1:]

        filename, file_extension = figurename.stem, figurename.suffix

        filename = filename + suffix

        if not file_extension:
            file_extension = "." + myformat

        if not figurename:
            figurename = "MyFigure" + file_extension
        else:
            figurename = filename + file_extension

        path = self.context.path
        figurename_fullpath = Path.joinpath(path, subfolderpath, figurename)
        full_subfolderpath = Path.joinpath(path, subfolderpath)
        if not Path.is_dir(full_subfolderpath):
            Path.mkdir(full_subfolderpath)
        print(f"Saving figure to {figurename_fullpath}")
        plt.savefig(
            figurename_fullpath,
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            papertype=None,
            format=file_extension[1:],
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )

    # Array viz helper functions
    def _make_2D_surface(
        self,
        fig,
        ax,
        data,
        x_values=None,
        y_values=None,
        x_label=None,
        y_label=None,
        variable_name=None,
        variable_unit=None,
        annotation=True,
        logscale=False,
        significance_annotation_labels=None,
    ):

        norm = None
        if logscale is True and not np.all(data == 0):
            from matplotlib.colors import LogNorm

            norm = LogNorm()
            # LogNorm does not understand 0 values and sets them bad, which makes them white.
            # Here we set them to the 0-value of the selected colormap
            cmap = get_cmap(self.cmap)
            cmap.set_bad(cmap(0.0))

        if annotation is True:
            if significance_annotation_labels is not None:
                res = sns.heatmap(
                    data,
                    cmap=self.cmap,
                    ax=ax,
                    cbar=True,
                    annot=significance_annotation_labels,
                    fmt="",
                    annot_kws={"color": "r"},
                    xticklabels=x_values,
                    yticklabels=y_values,
                    linewidths=0.5,
                    linecolor="black",
                    vmin=0.33,
                    vmax=1.0,
                )
                for t in res.texts:
                    if "*" in t.get_text():
                        t.set_text("*")
                    else:
                        t.set_text("")
            else:
                sns.heatmap(
                    data,
                    cmap=self.cmap,
                    ax=ax,
                    cbar=True,
                    annot=True,
                    fmt=".2g",
                    xticklabels=x_values,
                    yticklabels=y_values,
                    norm=norm,
                )
        else:
            sns.heatmap(
                data,
                cmap=self.cmap,
                ax=ax,
                cbar=True,
                annot=False,
                fmt=".2g",
                xticklabels=x_values,
                yticklabels=y_values,
                linewidths=0.5,
                linecolor="black",
                norm=norm,
            )

        # Set common labels
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)

    def _make_3D_surface(
        self,
        ax,
        x_values,
        y_values,
        z_values,
        logscale=None,
        x_label=None,
        y_label=None,
        variable_name=None,
    ):

        X, Y = np.meshgrid(x_values, y_values)
        dense_grid_steps = 50  # This should be relative to search space density
        assert (
            dense_grid_steps > x_values.shape[0]
        ), "Interpolation to less than original data, aborting..."  # You can comment this out and think it as a warning
        grid_x, grid_y = np.mgrid[
            np.min(x_values) : np.max(x_values) : dense_grid_steps * 1j,
            np.min(y_values) : np.max(y_values) : dense_grid_steps * 1j,
        ]
        values = z_values.flatten()
        points = np.array([X.flatten(), Y.flatten()]).T
        grid_z2 = griddata(
            points, values, (grid_x, grid_y), method="nearest"
        )  # , linear, nearest, cubic
        ax.plot_surface(grid_x, grid_y, grid_z2, ccount=50, rcount=50, cmap=self.cmap)

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if variable_name is not None:
            ax.set_zlabel(variable_name)

    def _make_table(self, ax, text_keys_list=[], text_values_list=[]):

        for i, (this_key, this_value) in enumerate(
            zip(text_keys_list, text_values_list)
        ):
            # ax.text(0.01, 0.9, f"{this_key}: {this_value}", va="top", ha="left")
            j = i - (i // 5) * 5
            ax.text(
                0.01 + (i // 5) * 0.3, 0.8 - (j * 0.15), f"{this_key}: {this_value}"
            )
            ax.tick_params(labelbottom=False, labelleft=False)

        ax.tick_params(
            axis="both",  # changes apply to both x and y axis; 'x', 'y', 'both'
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the left edge are off
            bottom=False,  # ticks along the bottom edge are off
            labelleft=False,  # labels along the left edge are off
            labelbottom=False,
        )

    def _prep_group_figure(self, NG_list):

        n_images = len(NG_list)
        if n_images == 1:
            n_columns = 1
        else:
            n_columns = 2

        n_rows = int(np.ceil(n_images / n_columns))

        fig, axs = plt.subplots(n_rows, n_columns)

        return fig, fig.axes

    def _prep_array_figure(self, two_dim):

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

        if two_dim:
            ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
            ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, projection="3d")
        else:
            ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2)

        return fig, fig.axes

    # Main visualization functions
    def show_readout_on_input(
        self,
        results_filename=None,
        normalize=False,
        unit_idx_list=None,
        savefigname=None,
    ):
        """
        Get input, get data. Scaling. turn to df, format df, Plot curves.
        """
        # Get data and input
        data = self.data_io.get_data(filename=results_filename, data_type="results")

        analog_signal = self.data_io.read_input_matfile(
            filename=self.context.input_filename, variable="stimulus"
        )

        NG_name = self.context.NG_name
        t = data["vm_all"][NG_name]["t"]  # All timesteps
        assert t.ndim == 1, "timepoints are not a 1-dim vector, aborting..."

        data_vm = data["vm_all"][NG_name]["vm"]

        self.data_is_valid(data_vm)
        self.data_is_valid(t)
        self.data_is_valid(analog_signal)

        if unit_idx_list is not None:
            # Get a subsample of units
            idx = np.asarray(unit_idx_list)
            analog_signal = analog_signal[:, idx]
            data_vm = data_vm[:, idx]

        if normalize == True:
            analog_signal = (analog_signal - np.min(analog_signal)) / np.ptp(
                analog_signal
            )

        # Create dict and column for timepoints
        id_var = "t"
        data_dict = {id_var: t}
        Ntimepoints = t.shape[0]

        # columns for input
        data_dict_in, dims_IN = self._build_columns(
            data_dict, analog_signal, Ntimepoints, "IN"
        )

        # columns for vm
        data_dict_vm, dims_Dec_vm = self._build_columns(
            data_dict, data_vm, Ntimepoints, "Dec_vm"
        )

        # # Get final output dimension, to get values for unpivot
        # prod_dims = dims_IN * dims_Dec_vm

        df_from_arr_in = pd.DataFrame(data=data_dict_in)
        df_from_arr_vm = pd.DataFrame(data=data_dict_vm)

        nsamples = len(df_from_arr_in.index)
        start_idx = self.context.t_idx_start
        end_idx = self.end2idx(self.context.t_idx_end, nsamples)
        df_from_arr_in = self._get_cut_data(
            df_from_arr_in, start_idx, end_idx, nsamples
        )
        df_from_arr_vm = self._get_cut_data(
            df_from_arr_vm, start_idx, end_idx, nsamples
        )

        value_vars_in = df_from_arr_in.columns[df_from_arr_in.columns != id_var]
        value_vars_vm = df_from_arr_vm.columns[df_from_arr_vm.columns != id_var]
        df_from_arr_unpivot_in = pd.melt(
            df_from_arr_in,
            id_vars=[id_var],
            value_vars=value_vars_in,
            var_name="units_in",
            value_name="data_in",
        )

        df_from_arr_unpivot_vm = pd.melt(
            df_from_arr_vm,
            id_vars=[id_var],
            value_vars=value_vars_vm,
            var_name="units_vm",
            value_name="data_vm",
        )

        fig, ax1 = plt.subplots(1, 1)
        sns.lineplot(
            x="t",
            y="data_in",
            data=df_from_arr_unpivot_in,
            hue="units_in",
            palette="dark",
            ax=ax1,
        )
        plt.legend(loc="upper left")
        ax2 = plt.twinx()
        sns.lineplot(
            x="t",
            y="data_vm",
            data=df_from_arr_unpivot_vm,
            palette="bright",
            hue="units_vm",
            ax=ax2,
        )
        plt.legend(loc="upper right")

        if normalize == True:
            EL = data["Neuron_Groups_Parameters"][NG_name]["namespace"]["EL"]
            VT = data["Neuron_Groups_Parameters"][NG_name]["namespace"]["VT"]
            plt.ylim(EL, VT)

        if savefigname:
            self.figsave(figurename=savefigname)

    def _show_coherence_of_two_signals(
        self,
        f,
        Cxy,
        Pwelch_spec_x,
        Pwelch_spec_y,
        Pxy,
        lags,
        corr,
        latency,
        x,
        y,
        x_scaled,
        y_scaled,
    ):

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

        # ax1.semilogy(f,Cxy)
        ax1.plot(f, Cxy)
        ax1.set_xlabel("frequency [Hz]")
        ax1.set_ylabel("Coherence")
        ax1.set_title("Coherence")

        self._string_on_plot(
            ax1, variable_name="Sum", variable_value=Cxy.sum(), variable_unit="a.u."
        )

        # ax2.semilogy(f,Pwelch_spec_x/np.max(Pwelch_spec_x))
        ax2.plot(f, Pwelch_spec_x / np.max(Pwelch_spec_x))
        # ax2.semilogy(f,Pwelch_spec_y/np.max(Pwelch_spec_y))
        ax2.plot(f, Pwelch_spec_y / np.max(Pwelch_spec_y))
        ax2.set_xlabel("frequency [Hz]")
        ax2.set_ylabel("PSD")
        ax2.set_title("Scaled power spectra")

        # ax3.semilogy(f,np.abs(Pxy))
        ax3.plot(f, np.abs(Pxy))
        ax3.set_xlabel("frequency [Hz]")
        ax3.set_ylabel("CSD [V**2/Hz]")
        ax3.set_title("Cross spectral density")

        ax4.plot(lags, corr)
        ax4.set_xlabel("samples")
        ax4.set_ylabel("correlation")
        ax4.set_title("Cross correlation")

        ax5.plot(x)
        ax5.plot(y)
        ax5.set_xlabel("time (samples)")
        ax5.set_ylabel("signal")
        ax5.set_title("Original signals")

        ax6.plot(x_scaled)
        ax6.plot(y_scaled)
        ax6.set_xlabel("time (samples)")
        ax6.set_ylabel("signal")
        ax6.set_title("Zero mean - unit variance signals")

        self._string_on_plot(
            ax4, variable_name="Latency", variable_value=latency, variable_unit="ms"
        )

    def _get_tricolormaps(self, idx, best_is="max", my_threshold=0.5):

        main_colors = ["red", "green", "blue"]
        cdict = {}
        target_color = main_colors[idx]

        for this_color in main_colors:
            if best_is == "max":
                if this_color == target_color:
                    cdict[this_color] = [
                        (0.0, 1.0, 1.0),
                        (my_threshold, 1.0, 1.0),
                        (1.0, 1.0, 1.0),
                    ]
                else:
                    cdict[this_color] = [
                        (0.0, 1.0, 1.0),
                        (my_threshold, 1.0, 1.0),
                        (1.0, 0.0, 0.0),
                    ]
                cdict["alpha"] = [
                    (0.0, 0.0, 0.0),
                    (my_threshold, 0.0, 0.0),
                    (1.0, 1.0, 1.0),
                ]
            elif best_is == "min":
                if this_color == target_color:
                    cdict[this_color] = [
                        (0.0, 1.0, 1.0),
                        (my_threshold, 1.0, 1.0),
                        (1.0, 1.0, 1.0),
                    ]
                else:
                    cdict[this_color] = [
                        (0.0, 0.0, 0.0),
                        (my_threshold, 1.0, 1.0),
                        (1.0, 1.0, 1.0),
                    ]
                cdict["alpha"] = [
                    (0.0, 1.0, 1.0),
                    (my_threshold, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                ]

        my_cmap = LinearSegmentedColormap("my_cmap1", cdict, N=256, gamma=1.0)

        return my_cmap

    def show_spikes(self, results_filename=None, savefigname=""):

        data = self.data_io.get_data(filename=results_filename, data_type="results")

        # Visualize
        # Extract connections from data dict
        NG_list = [n for n in data["spikes_all"].keys() if "NG" in n]

        print(NG_list)
        fig, axs = self._prep_group_figure(NG_list)

        for ax, this_group in zip(axs, NG_list):

            plot_data = (
                data["spikes_all"][this_group]["t"],
                data["spikes_all"][this_group]["i"],
            )

            for this_array in plot_data:
                self.data_is_valid(this_array, accept_empty=True)

            im = ax.plot(
                data["spikes_all"][this_group]["t"],
                data["spikes_all"][this_group]["i"],
                ".",
            )
            ax.set_title(this_group, fontsize=10)

            MeanFR = self.ana._analyze_meanfr(data, this_group)
            self._string_on_plot(
                ax, variable_name="Mean FR", variable_value=MeanFR, variable_unit="Hz"
            )

        if savefigname:
            self.figsave(figurename=savefigname)

    def show_analog_results(
        self,
        results_filename=None,
        savefigname="",
        param_name=None,
        startswith=None,
        neuron_index=None,
    ):
        # Shows data on filename. If filename remains None, shows the most recent data.

        assert param_name is not None, "Parameter param_name not defined, aborting..."
        assert (
            startswith is not None
        ), 'Parameter startswith not defined. Use "NG" or "S", aborting...'
        data = self.data_io.get_data(filename=results_filename, data_type="results")

        # Visualize
        assert (
            f"{param_name}_all" in data.keys()
        ), f"No {param_name} data found. Was it recorded? Aborting..."
        group_list = [
            n for n in data[f"{param_name}_all"].keys() if n.startswith(f"{startswith}")
        ]
        print(group_list)

        t = data[f"{param_name}_all"][group_list[0]]["t"]
        time_array = t / t.get_best_unit()
        fig, axs = self._prep_group_figure(group_list)

        for ax, this_group in zip(axs, group_list):

            N_monitored_neurons, this_data = self._get_n_neurons_and_data_array(
                data, this_group, param_name, neuron_index=neuron_index
            )

            if hasattr(this_data, "get_best_unit"):
                data_array = this_data / this_data.get_best_unit()
                this_unit = this_data.get_best_unit()
            else:
                data_array = this_data
                this_unit = "1"

            self.data_is_valid(data_array, accept_empty=False)

            im = ax.plot(time_array, data_array)
            ax.set_title(this_group, fontsize=10)
            ax.set_xlabel(t.get_best_unit())
            ax.set_ylabel(this_unit)

        fig.suptitle(f"{param_name}", fontsize=16)

        if savefigname:
            self.figsave(figurename=savefigname)

    def show_currents(self, results_filename=None, savefigname="", neuron_index=None):

        data = self.data_io.get_data(results_filename, data_type="results")
        # Visualize
        # Extract connections from data dict
        list_of_results_ge = [n for n in data["ge_soma_all"].keys() if "NG" in n]
        list_of_results_gi = [n for n in data["gi_soma_all"].keys() if "NG" in n]
        list_of_results_vm = [n for n in data["vm_all"].keys() if "NG" in n]

        assert (
            list_of_results_ge == list_of_results_gi == list_of_results_vm
        ), "Some key results missing, aborting..."
        NG_list = list_of_results_ge

        t = data["ge_soma_all"][NG_list[0]]["t"]

        fig, axs = self._prep_group_figure(NG_list)

        t_idx_start = self.context.t_idx_start
        t_idx_end = self.context.t_idx_end
        for ax, NG in zip(axs, NG_list):

            I_e, I_i, I_leak = self.get_currents_by_interval(
                data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end
            )

            if neuron_index is None:
                # print(f'Showing mean current over all neurons for group {NG}')
                N_neurons = I_e.shape[1]
                I_e_display = I_e.sum(axis=1) / N_neurons
                I_i_display = I_i.sum(axis=1) / N_neurons
            else:
                I_e_display = I_e[:, neuron_index[NG]]
                I_i_display = I_i[:, neuron_index[NG]]

            # Inhibitory current is negative, invert to cover same space as excitatory current
            I_i_display = I_i_display * -1

            ax.plot(t[t_idx_start:t_idx_end], np.array([I_e_display, I_i_display]).T)

            ax.legend(["I_e", "I_i"])
            ax.set_title(NG + " current", fontsize=10)

        if savefigname:
            self.figsave(figurename=savefigname)

    def show_conn(self, conn_file=None, hist_from=None, savefigname=""):

        data = self.data_io.get_data(filename=conn_file, data_type="connections")

        # Visualize
        # Extract connections from data dict
        conn_list = [n for n in data.keys() if "__to__" in n]

        # Pick histogram data
        if hist_from is None:
            hist_from = conn_list[-1]

        print(conn_list)
        fig, axs = self._prep_group_figure(conn_list)

        for ax, conn in zip(axs, conn_list):
            im = ax.imshow(data[conn]["data"].todense())
            ax.set_title(conn, fontsize=10)
            fig.colorbar(im, ax=ax)
        data4hist = np.squeeze(np.asarray(data[hist_from]["data"].todense().flatten()))
        data4hist_nozeros = np.ma.masked_equal(data4hist, 0)
        Nzeros = data4hist == 0
        proportion_zeros = Nzeros.sum() / Nzeros.size

        self.data_is_valid(data4hist_nozeros, accept_empty=False)

        n_images = len(conn_list)
        n_rows = int(np.ceil(n_images / 2))
        axs[(n_rows * 2) - 1].hist(data4hist_nozeros)
        axs[(n_rows * 2) - 1].set_title(
            f"{hist_from}\n{(proportion_zeros * 100):.1f}% zeros (not shown)"
        )
        if savefigname:
            self.figsave(figurename=savefigname)

    def _ascending_order_according_to_values(self, data_df, value_column_name, two_dim):
        """
        Internal function to sort dataframe according to values inside a string.
        """

        if not two_dim:

            # Get columns whose names contain both "Dimension" and "Value"
            value_column_df = data_df[
                [c for c in data_df.columns if "Value" in c and "Dimension" in c]
            ]
            value_column_name = value_column_df.columns[0]

            # Check is the value columns contain string instance
            if value_column_df.dtypes[0] == object:
                # If so, strip the string and convert to float
                def strip_characters(input_string):
                    import re

                    numeric_string = re.sub("[^0-9]", "", input_string)
                    return numeric_string

                value_column_df = value_column_df.applymap(strip_characters)

                # Get the longest string
                max_string_length = value_column_df.apply(
                    lambda x: len(x.values[0])
                ).max()

                # Pad the strings with zeros to the same length
                value_column_df = value_column_df.applymap(
                    lambda x: x.zfill(max_string_length)
                )

                # Sort by numerical values inside strings
                value_column_df = value_column_df.sort_values(by=value_column_name)

                # Apply the same sort order to data_df
                data_df = data_df.reindex(value_column_df.index)

                # Reset index
                data_df.reset_index(drop=True, inplace=True)

        return data_df

    def show_analyzed_arrayrun(
        self,
        csv_filename=None,
        analysis=None,
        variable_unit=None,
        NG_id_list=[],
        annotation_2D=True,
        logscale=False,
        annotate_with_p=False,
    ):
        """
        Pseudocode
        Get MeanFR_TIMESTAMP_.csv
        If does not exist, calculate from metadata file list
        Prep figure in subfunction, get axes handles
        Table what is necessary, display
        Plot 2D
        Plot 3D
        """
        if analysis is not None:
            assert (
                analysis.lower() in self.map_ana_names.keys()
            ), "Unknown analysis, aborting..."

            analysisHR = self.map_ana_names[analysis.lower()]

        # Get data from analysisHR_TIMESTAMP_.csv
        try:
            if csv_filename is None:
                data_df = self.data_io.get_data(data_type=analysisHR)
            elif csv_filename[-4:] == ".csv":
                data_df = self.data_io.get_data(filename=csv_filename, data_type=None)
            else:
                data_df = self.data_io.get_data(
                    filename=csv_filename + ".csv", data_type=None
                )
        # If does not exist, calculate from metadata file list
        except FileNotFoundError as error:
            print(error)
            print(
                "Conducting necessary analysis first. Using most recent metadata file and full duration"
            )
            self.analyze_arrayrun(analysis=analysisHR)
            data_df = self.data_io.get_data(data_type=analysisHR)

        if analysisHR.lower() in ["meanfr", "meanvm", "eicurrentdiff"]:
            print(f"Creating one figure for each neuron group")
        elif analysisHR.lower() in ["grcaus", "classify", "economy"]:
            print(f"Creating one figure for each analysis")

        # Below we expect the analysis name to be part of column name.
        # This works as substring to pick up the correct dependent variables
        analyzes_for_zipping = [analysisHR] * len(data_df.columns)
        available_data_column_list = [
            ng
            for (dtype, ng) in zip(analyzes_for_zipping, data_df.columns)
            if dtype.lower() in ng.lower()
        ]

        if not NG_id_list:
            print("All neuron groups requested")
            requested_data_column_list = available_data_column_list
        else:
            requested_data_column_list = []
            for this_NG_id in NG_id_list:
                for this_data_column in available_data_column_list:
                    if this_NG_id in this_data_column:
                        requested_data_column_list.append(this_data_column)

        NG_id_list = []
        NG_name_list = []

        for this_data_column in requested_data_column_list:
            start_idx = this_data_column.find("NG")
            end_idx = this_data_column.find("_", start_idx)
            NG_id_list.append(this_data_column[start_idx:end_idx])
            uscore_idx = this_data_column.find("_", end_idx) + 1
            NG_name_list.append(this_data_column[uscore_idx:])

        if "Dimension-2 Parameter" in data_df.columns:
            two_dim = True
        else:
            two_dim = False

        if analysisHR.lower() in ["grcaus"]:
            variable_unit_dict = {
                "_p": "p value",
                "_Information": "(bits)",
                "_latency": "latency (s)",
                "_TransfEntropy": "Transfer Entropy (bits/sample)",
                "_isStationary": "boolean",
                "_TargetEntropy": "target signal entropy (bits)",
                "_fitQuality": "mean fit quality",
            }

        for this_NG_id, this_NG_name, this_data_column in zip(
            NG_id_list, NG_name_list, requested_data_column_list
        ):

            assert (
                this_NG_id in this_data_column
            ), "Neuron group does not match data column, aborting ..."

            if "variable_unit_dict" in locals():
                for this_key in variable_unit_dict.keys():
                    if this_key in this_NG_name:
                        variable_unit = variable_unit_dict[this_key]

            # Prep figure in subfunction, get axes handles
            fig, axs = self._prep_array_figure(two_dim)

            # Table what is necessary, display
            text_keys_list = [
                "Analysis",
                "Neuron Group #",
                "Neuron Group Name",
                "MIN value - (y,x)",
                "MAX value - (y,x)",
                "MIN at Params",
                "MAX at Params",
            ]
            text_values_list = []
            text_values_list.append(analysisHR)
            text_values_list.append(this_NG_id)
            text_values_list.append(this_NG_name)

            value_column_name = this_data_column
            variable_name = analysisHR

            significance_annotation_labels = None
            if annotation_2D is True:
                if annotate_with_p:
                    # A preparation for annotation asterisks on sns heatmap
                    if "accuracy" in value_column_name:
                        p_value_column_name = value_column_name.replace("accuracy", "p")
                        if two_dim:
                            (
                                p_values,
                                x_label,
                                y_label,
                                df_2d,
                                x_values,
                                y_values,
                                min_value,
                                min_idx,
                                max_value,
                                max_idx,
                            ) = self.ana.get_df_extremes(
                                data_df, p_value_column_name, two_dim=two_dim
                            )
                        else:
                            (
                                p_values,
                                x_label,
                                x_values,
                                min_value,
                                min_idx,
                                max_value,
                                max_idx,
                            ) = self.ana.get_df_extremes(
                                data_df, p_value_column_name, two_dim=two_dim
                            )
                        alpha = 0.05
                        bonferroni_level = alpha / p_values.size
                        bonferroni_corrected_significant_bool = (
                            p_values < bonferroni_level
                        )
                        asterisk_array = np.chararray(
                            bonferroni_corrected_significant_bool.shape
                        )
                        asterisk_array[:] = ""
                        asterisk_array[bonferroni_corrected_significant_bool] = "*"
                        significance_annotation_labels = asterisk_array
                    else:
                        print(
                            "\nINFO: 'accuracy' not found in value_column_name, cannot annotate_with_p, continuing..."
                        )

            # Use this to manually order according to values inside strings. Use for arrays where string, such as file name, contain a running number
            if 0:
                data_df = self._ascending_order_according_to_values(
                    data_df, value_column_name, two_dim
                )

            if two_dim:
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
                ) = self.ana.get_df_extremes(
                    data_df, value_column_name, two_dim=two_dim
                )
            else:
                (
                    data_nd_array,
                    x_label,
                    x_values,
                    min_value,
                    min_idx,
                    max_value,
                    max_idx,
                ) = self.ana.get_df_extremes(
                    data_df, value_column_name, two_dim=two_dim
                )

            min_value_rounded = self.round_to_n_significant(
                min_value, significant_digits=3
            )
            text_values_list.append(f"{min_value_rounded} {variable_unit}- {min_idx}")
            max_value_rounded = self.round_to_n_significant(
                max_value, significant_digits=3
            )
            text_values_list.append(f"{max_value_rounded} {variable_unit} - {max_idx}")

            if two_dim:
                text_values_list.append(
                    f"{y_label} = {df_2d.index[min_idx[0]]}; {x_label} = {df_2d.columns[min_idx[1]]}"
                )
                text_values_list.append(
                    f"{y_label} = {df_2d.index[max_idx[0]]}; {x_label} = {df_2d.columns[max_idx[1]]}"
                )
            else:
                text_values_list.append(
                    f'{x_label} = {data_df["Dimension-1 Value"][min_idx[0]]}'
                )
                text_values_list.append(
                    f'{x_label} = {data_df["Dimension-1 Value"][max_idx[0]]}'
                )

            self._make_table(
                axs[0], text_keys_list=text_keys_list, text_values_list=text_values_list
            )

            self.data_is_valid(data_nd_array, accept_empty=False)

            if two_dim:
                self._make_2D_surface(
                    fig,
                    axs[1],
                    data_nd_array,
                    x_values=x_values,
                    y_values=y_values,
                    logscale=logscale,
                    x_label=x_label,
                    y_label=y_label,
                    variable_name=variable_name,
                    variable_unit=variable_unit,
                    annotation=annotation_2D,
                    significance_annotation_labels=significance_annotation_labels,
                )

                self._make_3D_surface(
                    axs[2],
                    x_values,
                    y_values,
                    data_nd_array,
                    logscale=logscale,
                    x_label=x_label,
                    y_label=y_label,
                    variable_name=variable_name,
                )
            else:
                axs[1].plot(x_values, data_nd_array)
                axs[1].set_xlabel(f"{x_label}")
                axs[1].set_ylabel(f"{variable_name} ({variable_unit})")

            if self.save_figure_with_arrayidentifier is not None:
                # Assuming the last word in this_NG_name contains the necessary id
                suffix_start_idx = this_NG_name.rfind("_")
                id = this_NG_name[suffix_start_idx + 1 :]
                if analysisHR.lower() in ["meanfr", "meanvm", "eicurrentdiff"]:
                    id = this_NG_id + this_data_column[this_data_column.rfind("_") :]
                elif id == analysisHR.lower():
                    id = this_data_column[: this_data_column.find("_")]

                self.figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{analysisHR}_{id}",
                    myformat="svg",
                    subfolderpath=self.save_figure_to_folder,
                )
                self.figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{analysisHR}_{id}",
                    myformat="png",
                    subfolderpath=self.save_figure_to_folder,
                )

    def show_input_to_readout_coherence(
        self, results_filename=None, savefigname="", signal_pair=[0, 0]
    ):

        data_dict = self.data_io.get_data(
            filename=results_filename, data_type="results"
        )

        analog_input = self.data_io.get_data(
            self.context.input_filename, data_type=None
        )
        source_signal = analog_input["stimulus"].T  # We want time x units

        NG = [n for n in data_dict["vm_all"].keys() if "NG3" in n]
        vm_unit = self.cxparser.get_vm_by_interval(data_dict, NG[0])
        target_signal = vm_unit / vm_unit.get_best_unit()

        x = source_signal[
            self.context.t_idx_start : self.context.t_idx_end, signal_pair[0]
        ]
        y = target_signal[
            self.context.t_idx_start : self.context.t_idx_end, signal_pair[1]
        ]

        high_cutoff = 100  # Frequency in Hz
        nsamples = self.cxparser.get_n_samples(data_dict)
        nperseg = nsamples // 6
        dt = self.cxparser.get_dt(data_dict)
        samp_freq = 1.0 / dt  # 1.0/(dt * downsampling_factor)

        self.data_is_valid(x, accept_empty=False)
        self.data_is_valid(y, accept_empty=False)

        (
            f,
            Cxy,
            Pwelch_spec_x,
            Pwelch_spec_y,
            Pxy,
            lags,
            corr,
            coherence_sum,
            x_scaled,
            y_scaled,
        ) = self.ana.get_coherence_of_two_signals(
            x, y, samp_freq=samp_freq, nperseg=nperseg, high_cutoff=high_cutoff
        )
        shift_in_seconds = self.ana.get_cross_corr_latency(lags, corr, dt)
        shift_in_milliseconds = shift_in_seconds * 1000

        self._show_coherence_of_two_signals(
            f,
            Cxy,
            Pwelch_spec_x,
            Pwelch_spec_y,
            Pxy,
            lags,
            corr,
            shift_in_milliseconds,
            x,
            y,
            x_scaled,
            y_scaled,
        )

        if savefigname:
            self.figsave(figurename=savefigname)

    def show_estimate_on_input(
        self,
        results_filename=None,
        simulation_engine="cxsystem",
        readout_group="E",
        decoding_method="least_squares",
        output_type="estimated",
        unit_idx_list=[0],
    ):

        Error, xL, xest = self.ana.get_MSE(
            results_filename=results_filename,
            simulation_engine=simulation_engine,
            readout_group=readout_group,
            decoding_method=decoding_method,
            output_type=output_type,
        )

        self.data_is_valid(xL, accept_empty=False)
        self.data_is_valid(xest, accept_empty=False)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(xL[:, unit_idx_list])
        ax.plot(xest[:, unit_idx_list])

        self._string_on_plot(
            ax, variable_name="Error", variable_value=Error, variable_unit="a.u."
        )

        if len(unit_idx_list) == 1:
            plt.legend(["Target", "Estimate"])

    def system_polar_bar(self, row_selection=None, folder_name=None):

        if isinstance(row_selection, list) and len(row_selection) > 1:
            # For list of rows, make one figure for each row
            for this_row in row_selection:
                self.system_polar_bar(row_selection=this_row, folder_name=folder_name)
            return
        elif isinstance(row_selection, list) and len(row_selection) == 1:
            row_selection = row_selection[0]

        (
            data0_df,
            data_df_compiled,
            independent_var_col_list,
            dependent_var_col_list,
            time_stamp,
        ) = self.data_io.get_csv_as_df(folder_name=folder_name)

        # Combine dfs
        (
            df_for_barplot,
            profile_metrics_columns_list,
            min_values,
            max_values,
        ) = self.ana._get_system_profile_metrics(
            data_df_compiled, independent_var_col_list
        )

        # Normalize magnitudes
        values_np = df_for_barplot[
            profile_metrics_columns_list
        ].values  # returns a numpy array
        values_np_scaled = self.ana.scaler(
            values_np, scale_type="minmax", feature_range=[0, 1]
        )
        df_for_barplot[profile_metrics_columns_list] = values_np_scaled

        # extra points are the original dimensions, to be visualized
        extra_points = np.vstack(
            [
                np.eye(len(profile_metrics_columns_list)),
                -1 * np.eye(len(profile_metrics_columns_list)),
            ]
        )

        # Get PCA of data. Note option for extra_points=extra_points_df
        (
            values_pca,
            principal_axes_in_PC_space,
            explained_variance_ratio,
            extra_points_pca,
        ) = self.ana.get_PCA(
            values_np,
            n_components=2,
            col_names=profile_metrics_columns_list,
            extra_points=extra_points,
            extra_points_at_edge_of_gamut=True,
        )

        # Define PCA space
        xmin, xmax = np.min(values_pca[:, 0]), np.max(values_pca[:, 0])
        ymin, ymax = np.min(values_pca[:, 1]), np.max(values_pca[:, 1])
        PC0_limits = np.array([xmin - xmin / 10, xmax + xmax / 10])
        PC1_limits = np.array([ymin - ymin / 10, ymax + ymax / 10])

        PC_cardinal_points = np.array(
            [
                [PC0_limits[0], PC1_limits[0]],
                [PC0_limits[1], PC1_limits[1]],
                [PC0_limits[0], PC1_limits[1]],
                [PC0_limits[1], PC1_limits[0]],
            ]
        )

        # Number of parameters for the polar bar chart
        N = len(profile_metrics_columns_list)

        # Polar angle (in radians) of each parameter
        theta = np.linspace(2 * np.pi / N, 2 * np.pi, N, endpoint=True)

        # Magnitude as radius of bar in the polar plot
        if row_selection is None:
            row_selection = 0
        radii = df_for_barplot.iloc[row_selection, :][
            profile_metrics_columns_list
        ].values

        # Second dimension as width
        width = 2 * np.pi / N

        # Third dimension as colors
        profile_in_PC_coordinates = extra_points_pca
        profile_in_RGB_coordinates = self._pc2rgb(
            PC_cardinal_points, additional_points=profile_in_PC_coordinates
        )

        # colors = plt.cm.viridis(radii.astype(float))
        colors = profile_in_RGB_coordinates

        self.data_is_valid(PC_cardinal_points, accept_empty=False)
        self.data_is_valid(values_pca, accept_empty=False)
        self.data_is_valid(theta, accept_empty=False)
        self.data_is_valid(radii, accept_empty=False)

        ## Plotting##
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(221, projection="polar")
        ax2 = plt.subplot(222, projection=None)
        ax3 = plt.subplot(223, projection=None)
        ax4 = plt.subplot(224, projection=None)

        ax1.bar(theta, radii, width=width, bottom=0.0, color=colors)

        # Polar tick positions in radial coordinates and corresponding label strings
        ax1.set_xticks(np.linspace(2 * np.pi / (N), 2 * np.pi, N))
        ax1.set_xticklabels(profile_metrics_columns_list)

        # Set radial max value
        ax1.set_rmax(1.0)
        # Radial ticks
        ax1.set_rticks([0.25, 0.5, 0.75, 1.0])  # Set radial ticks
        ax1.set_rlabel_position(-11.25)  # Move radial labels away from plotted line

        # Subplot title; selected point
        parameter1name = (
            f"{data_df_compiled.loc[row_selection, independent_var_col_list[0]]}"
        )
        parameter1value = (
            f"{data_df_compiled.loc[row_selection, independent_var_col_list[1]]:.2f}"
        )
        if "Dimension-2 Parameter" in data0_df.columns:
            parameter2name = (
                f"{data_df_compiled.loc[row_selection, independent_var_col_list[2]]}"
            )
            parameter2value = f"{data_df_compiled.loc[row_selection, independent_var_col_list[3]]:.2f}"
            ax1.title.set_text(
                f"{parameter1name}: {parameter1value}\n{parameter2name}: {parameter2value}"
            )
            figout_names = (
                parameter1name.replace(" ", "_")
                + parameter1value.replace(".", "p")
                + "_"
                + parameter2name.replace(" ", "_")
                + parameter2value.replace(".", "p")
            )
        else:
            ax1.title.set_text(f"{parameter1name}: {parameter1value}")
            figout_names = parameter1name.replace(" ", "_") + parameter1value.replace(
                ".", "p"
            )

        # Legend in another subplot (too big for the same)
        min_values = min_values.apply(self.round_to_n_significant, args=(3,)).values
        max_values = max_values.apply(self.round_to_n_significant, args=(3,)).values
        bar_dict = {
            f"{name}: min {minn}, max {maxx}": color
            for name, minn, maxx, color in zip(
                profile_metrics_columns_list, min_values, max_values, colors
            )
        }
        labels = list(bar_dict.keys())
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=bar_dict[label]) for label in labels
        ]
        ax2.set_axis_off()
        ax2.legend(handles, labels, loc="center")

        # Plot data on PC space with colors from CIExy map
        RGB_values = self._pc2rgb(PC_cardinal_points, additional_points=values_pca)
        ax3.scatter(values_pca[:, 0], values_pca[:, 1], c=RGB_values)

        # Selected point
        ax3.scatter(
            values_pca[row_selection, 0],
            values_pca[row_selection, 1],
            c=RGB_values[row_selection],
            edgecolors="k",
            s=100,
        )

        # Original N dim space, projected on 2 dim PC space
        extra_point_markers = "s"
        n_metrics = len(profile_metrics_columns_list)

        # End points
        ax3.scatter(
            extra_points_pca[:n_metrics, 0],
            extra_points_pca[:n_metrics, 1],
            marker=extra_point_markers,
            c="k",
        )
        pc0_label = f"PC0 ({100*explained_variance_ratio[0]:.0f}%)"
        pc1_label = f"PC1 ({100*explained_variance_ratio[1]:.0f}%)"
        ax3.set_xlabel(pc0_label)
        ax3.set_ylabel(pc1_label)
        ax3.grid()

        ## Display colorscale
        x = np.linspace(PC0_limits[0], PC0_limits[1], num=100)
        y = np.linspace(PC1_limits[0], PC1_limits[1], num=100)
        X, Y = np.meshgrid(x, y)
        PC_space = np.vstack([X.flatten(), Y.flatten()]).T
        CIEonRGB_scaled = self._pc2rgb(PC_cardinal_points, additional_points=PC_space)

        # Reshape to 2D
        imsize = (len(X), len(Y), 3)
        CIEonRGB_image = np.reshape(CIEonRGB_scaled, imsize, order="C")
        # Get extent of existing 2 dim PC plot
        PC_ax_extent = [ax3.get_xlim(), ax3.get_ylim()]
        # Plot colorscape
        ax4.imshow(
            CIEonRGB_image,
            origin="lower",
            extent=[
                PC_ax_extent[0][0],
                PC_ax_extent[0][1],
                PC_ax_extent[1][0],
                PC_ax_extent[1][1],
            ],
        )
        ax4.grid()

        # Create plot pairs
        for this_point in np.arange(n_metrics):
            start_end_x = [
                extra_points_pca[this_point, 0],
                extra_points_pca[n_metrics + this_point, 0],
            ]
            start_end_y = [
                extra_points_pca[this_point, 1],
                extra_points_pca[n_metrics + this_point, 1],
            ]
            ax4.plot(start_end_x, start_end_y, "k--", lw=0.5)
        # # Start points
        # ax4.scatter(extra_points_pca[n_metrics:,0], extra_points_pca[n_metrics:,1], marker = extra_point_markers, c='w', edgecolors='k', s=20)
        # End points
        ax4.scatter(
            extra_points_pca[:n_metrics, 0],
            extra_points_pca[:n_metrics, 1],
            marker=extra_point_markers,
            facecolors="none",
            edgecolors="k",
        )
        ax4.title.set_text(
            f"Projections of the {str(len(profile_metrics_columns_list))} dimensions"
        )

        if self.save_figure_with_arrayidentifier is not None:
            self.figsave(
                figurename=f"{self.save_figure_with_arrayidentifier}_summary_{figout_names}",
                myformat="svg",
            )

    def _build_param_plot(self, coll_ana_df_in, param_plot_dict, to_mpa_dict_in):
        """
        Prepare for parametric plotting of multiple conditions.
        This method fetch all data to
        data_list, data_sub_list, and sub_data_list.
        The corresponding names are in
        outer_name_list and data_name_list. Sub name is always the same.
        """

        sub_col_name = ""

        # Isolate inner sub ana, if active
        if param_plot_dict["inner_sub"] is True:
            coll_sub_S = coll_ana_df_in.loc[param_plot_dict["inner_sub_ana"]]
            coll_ana_df = coll_ana_df_in.drop(
                index=param_plot_dict["inner_sub_ana"], inplace=False
            )
            analyzes_list = copy.deepcopy(to_mpa_dict_in["analyzes"])
            analyzes_list.remove(param_plot_dict["inner_sub_ana"])
            to_mpa_dict = {k: to_mpa_dict_in[k] for k in ["midpoints", "parameters"]}
            to_mpa_dict["analyzes"] = analyzes_list
            # Short ana name for sub to diff from data col names
            sub_col_name = coll_sub_S["ana_name_prefix"]
        else:
            to_mpa_dict = to_mpa_dict_in
            coll_ana_df = coll_ana_df_in

        [title] = to_mpa_dict[param_plot_dict["title"]]
        outer_list = to_mpa_dict[param_plot_dict["outer"]]
        inner_list = to_mpa_dict[param_plot_dict["inner"]]

        # if paths to data provided, take inner names from distinct list
        if param_plot_dict["inner_paths"] is True:
            inner_list = param_plot_dict["inner_path_names"]

        mid_idx = list(param_plot_dict.values()).index("midpoints")
        par_idx = list(param_plot_dict.values()).index("parameters")
        ana_idx = list(param_plot_dict.values()).index("analyzes")

        key_list = list(param_plot_dict.keys())

        # Create dict whose key is folder hierarchy and value is plot hierarchy
        hdict = {
            "mid": key_list[mid_idx],
            "par": key_list[par_idx],
            "ana": key_list[ana_idx],
        }

        data_list = []  # nested list, N items = N outer x N inner
        data_name_list = []  # nested list, N items = N outer x N inner
        data_sub_list = []  # nested list, N items = N outer x N inner
        outer_name_list = []  # list , N items = N outer

        for outer in outer_list:
            inner_data_list = []  # list , N items = N inner
            inner_name_list = []  # list , N items = N inner
            inner_sub_data_list = []  # list , N items = N inner

            for in_idx, inner in enumerate(inner_list):
                # Nutcracker. eval to "outer", "inner" and "title"
                mid = eval(f"{hdict['mid']}")
                par = eval(f"{hdict['par']}")
                this_folder = f"{mid}_{par}"
                this_ana = eval(f"{hdict['ana']}")

                this_ana_col = coll_ana_df.loc[this_ana]["csv_col_name"]
                if param_plot_dict["compiled_results"] is True:
                    this_folder = f"{this_folder}_compiled_results"
                    this_ana_col = f"{this_ana_col}_mean"

                if param_plot_dict["inner_paths"] is True:
                    csv_path_tuple = param_plot_dict["paths"][in_idx]
                    csv_path = reduce(
                        lambda acc, y: Path(acc).joinpath(y), csv_path_tuple
                    )
                else:
                    csv_path = None

                # get data
                (
                    data0_df,
                    data_df_compiled,
                    independent_var_col_list,
                    dependent_var_col_list,
                    time_stamp,
                ) = self.data_io.get_csv_as_df(
                    folder_name=this_folder, csv_path=csv_path, include_only=None
                )

                df = data_df_compiled[this_ana_col]
                inner_data_list.append(df)
                inner_name_list.append(inner)

                if param_plot_dict["inner_sub"] is True:
                    sub_col = coll_sub_S["csv_col_name"]
                    if param_plot_dict["compiled_results"] is True:
                        sub_col = f"{sub_col}_mean"
                    df_sub = data_df_compiled[sub_col]
                    inner_sub_data_list.append(df_sub)

            data_list.append(inner_data_list)
            data_name_list.append(inner_name_list)
            data_sub_list.append(inner_sub_data_list)
            outer_name_list.append(outer)

        return (
            data_list,
            data_name_list,
            data_sub_list,
            outer_name_list,
            sub_col_name,
        )

    def show_catplot(self, param_plot_dict):

        """
        Visualization of parameter values in different categories. Data is collected in _build_param_plot, and all plotting is here.

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

        inner_paths : bool (only inner available for setting paths). Provide comparison from arbitrary paths, e.g. controls
        paths : provide list of tuples of full path parts to data folder.
        E.g. [(root_path, 'Single_narrow_iterations_control', 'Bacon_gL_compiled_results'),]
        The number of paths MUST be the same as the number of corresponding inner variables.

        compiled_results : data at compiled_results folder, mean over iterations
        param_plot_dict = {
            "title": "parameters",  # multiple allowed => each in separate figure
            "outer":  "analyzes",  # multiple allowed => plt subplot panels
            "inner": "midpoints",  # multiple allowed => direct comparison
            "inner_sub": False,  # A singular analysis => subdivisions
            "inner_sub_ana": "Excitatory Firing Rate",  #  The singular analysis
            "bin_edges": [[0.001, 150], [150, 300]],
            "plot_type": "box", # "violin",  "box", "strip", "swarm", "boxen", "point", "bar"
            "compiled_results": True,
            'sharey' : False,
            "inner_paths" : False,
            "paths": [],
        }
        """

        coll_ana_df = copy.deepcopy(self.coll_mpa_dict["coll_ana_df"])
        to_mpa_dict = copy.deepcopy(self.context.to_mpa_dict)

        titles = to_mpa_dict[param_plot_dict["title"]]

        if param_plot_dict["save_description"] is True:
            describe_df_list = []
            describe_df_columns_list = []
            describe_folder_full = Path.joinpath(self.context.path, "Descriptions")
            describe_folder_full.mkdir(parents=True, exist_ok=True)

        # Recursive call for multiple titles => multiple figures
        for this_title in titles:
            this_title_list = [this_title]
            to_mpa_dict[param_plot_dict["title"]] = this_title_list

            (
                data_list,
                data_name_list,
                data_sub_list,
                outer_name_list,
                sub_col_name,
            ) = self._build_param_plot(coll_ana_df, param_plot_dict, to_mpa_dict)

            sharey = param_plot_dict["sharey"]
            palette = param_plot_dict["palette"]

            if param_plot_dict["display_optimal_values"] is True:
                optimal_value_foldername = param_plot_dict["optimal_value_foldername"]
                optimal_description_name = param_plot_dict["optimal_description_name"]

                # read optimal values to dataframe from path/optimal_values/optimal_unfit_description.csv
                optimal_df = pd.read_csv(
                    Path.joinpath(
                        self.context.path,
                        optimal_value_foldername,
                        optimal_description_name,
                    )
                )
                # set the first column as index
                optimal_df.set_index(optimal_df.columns[0], inplace=True)

            fig, [axs] = plt.subplots(1, len(data_list), sharey=sharey, squeeze=False)

            for out_idx, inner_data_list in enumerate(data_list):
                outer_name = outer_name_list[out_idx]
                inner_df_coll = pd.DataFrame()
                sub_df_coll = pd.DataFrame()
                for in_idx, inner_df in enumerate(inner_data_list):
                    inner_df_coll[data_name_list[out_idx][in_idx]] = inner_df
                    if param_plot_dict["inner_sub"] is True:
                        sub_df_coll[data_name_list[out_idx][in_idx]] = data_sub_list[
                            out_idx
                        ][in_idx]

                self.data_is_valid(inner_df_coll.values, accept_empty=False)

                # Temporary fix 220421 SV
                if outer_name == "Coherence":
                    inner_df_coll = inner_df_coll / 34
                if param_plot_dict["save_description"] is True:
                    describe_df_list.append(inner_df_coll)  # for saving
                    describe_df_columns_list.append(f"{outer_name}")

                # We use axes level plots instead of catplot which is figure level plot.
                # This way we can control the plotting order and additional arguments
                if param_plot_dict["inner_sub"] is False:
                    # wide df--each column plotted
                    boxprops = dict(
                        linestyle="-", linewidth=1, edgecolor="black", facecolor=".7"
                    )

                    if param_plot_dict["plot_type"] == "box":
                        g1 = sns.boxplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            boxprops=boxprops,
                            whis=[0, 100],
                            showfliers=False,
                            showbox=True,
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "violin":
                        g1 = sns.violinplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "strip":
                        g1 = sns.stripplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "swarm":
                        g1 = sns.swarmplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "boxen":
                        g1 = sns.boxenplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "point":
                        g1 = sns.pointplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "bar":
                        g1 = sns.barplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )

                elif param_plot_dict["inner_sub"] is True:

                    inner_df_id_vars = pd.DataFrame().reindex_like(inner_df_coll)
                    # Make a “long-form” DataFrame
                    for this_bin_idx, this_bin_limits in enumerate(
                        param_plot_dict["bin_edges"]
                    ):

                        # Apply bin edges to sub data
                        inner_df_id_vars_idx = sub_df_coll.apply(
                            lambda x: (x > this_bin_limits[0])
                            & (x < this_bin_limits[1]),
                            raw=True,
                        )
                        inner_df_id_vars[inner_df_id_vars_idx] = this_bin_idx

                    inner_df_id_values_vars = pd.concat(
                        [
                            inner_df_coll.stack(dropna=False),
                            inner_df_id_vars.stack(dropna=False),
                        ],
                        axis=1,
                    )

                    inner_df_id_values_vars = inner_df_id_values_vars.reset_index()
                    inner_df_id_values_vars.drop(columns="level_0", inplace=True)
                    inner_df_id_values_vars.columns = [
                        "Title",
                        outer_name,
                        sub_col_name,
                    ]

                    bin_legends = [
                        f"{m}-{n}" for [m, n] in param_plot_dict["bin_edges"]
                    ]
                    inner_df_id_values_vars[sub_col_name].replace(
                        to_replace=[*range(0, len(param_plot_dict["bin_edges"]))],
                        value=bin_legends,
                        inplace=True,
                    )

                    if param_plot_dict["plot_type"] == "box":
                        g1 = sns.boxplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            whis=[0, 100],
                            showfliers=False,
                            showbox=True,
                        )
                    elif param_plot_dict["plot_type"] == "violin":
                        g1 = sns.violinplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            split=True,
                        )
                    elif param_plot_dict["plot_type"] == "strip":
                        g1 = sns.stripplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "swarm":
                        g1 = sns.swarmplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "boxen":
                        g1 = sns.boxenplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "point":
                        g1 = sns.pointplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "bar":
                        g1 = sns.barplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )

                g1.set(xlabel=None, ylabel=None)
                fig.suptitle(this_title, fontsize=16)

                labels = data_name_list[out_idx]
                axs[out_idx].set_xticklabels(labels, rotation=60)

                if param_plot_dict["display_optimal_values"] is True:
                    # Get column name from coll_ana_df
                    col_name = coll_ana_df.loc[outer_name, "csv_col_name"]
                    matching_column = [
                        c
                        for c in optimal_df.columns
                        if c.startswith(col_name) and c.endswith("_mean")
                    ]
                    if len(matching_column) > 0:
                        min_value = optimal_df.loc["min", matching_column[0]]
                        max_value = optimal_df.loc["max", matching_column[0]]
                        # draw a horizontal dashed line to axs[out_idx] at y=min_value and y=max_value
                        axs[out_idx].axhline(y=min_value, color="black", linestyle="--")
                        axs[out_idx].axhline(y=max_value, color="black", linestyle="--")

                # If statistics is tested, set statistics value and name to each axs subplot
                if param_plot_dict["inner_stat_test"] is True:
                    """
                    Apply the statistical test to inner_df_coll
                    If len(inner_data_list) == 2, apply Wilcoxon signed-rank test.
                    Else if len(inner_data_list) > 2, apply Friedman test.
                    Set stat_name to the test name.
                    """
                    if len(inner_data_list) == 2:
                        stat_test_name = "Wilcoxon signed-rank test"
                        statistics, stat_p_value = self.ana.stat_tests.wilcoxon_test(
                            inner_df_coll.values[:, 0], inner_df_coll.values[:, 1]
                        )
                    elif len(inner_data_list) > 2:
                        stat_test_name = "Friedman test"
                        statistics, stat_p_value = self.ana.stat_tests.friedman_test(
                            inner_df_coll.values
                        )
                    else:
                        raise ValueError(
                            "len(inner_data_list) must be 2 or more for stat_test, aborting..."
                        )

                    stat_corrected_p_value = stat_p_value * len(
                        data_list
                    )  # Bonferroni correction

                    axs[out_idx].set_title(
                        f"{outer_name}\n{stat_test_name} =\n{stat_corrected_p_value:.3f}\nN = {inner_df_coll.shape[0]}"
                    )
                else:
                    axs[out_idx].set_title(outer_name)

            if param_plot_dict["save_description"] is True:
                describe_df_columns_list = [
                    c.replace(" ", "_") for c in describe_df_columns_list
                ]
                describe_df_all = pd.DataFrame()
                for this_idx, this_column in enumerate(describe_df_columns_list):
                    # Append the describe_df_all data describe_df_list[this_idx]
                    this_describe_df = describe_df_list[this_idx]
                    # Prepend the column names with this_column
                    this_describe_df.columns = [
                        this_column + "_" + c for c in this_describe_df.columns
                    ]

                    describe_df_all = pd.concat(
                        [describe_df_all, this_describe_df], axis=1
                    )

                filename_full = Path.joinpath(
                    describe_folder_full,
                    param_plot_dict["save_name"] + "_" + this_title + ".csv",
                )

                # Save the describe_df_all dataframe .to_csv(filename_full, index=False)
                describe_df_all_df = describe_df_all.describe()
                describe_df_all_df.insert(
                    0, "description", describe_df_all.describe().index
                )
                describe_df_all_df.to_csv(filename_full, index=False)
                describe_df_list = []
                describe_df_columns_list = []

            if self.save_figure_with_arrayidentifier is not None:

                id = "box"

                self.figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}_{this_title}",
                    myformat="svg",
                    subfolderpath=self.save_figure_to_folder,
                )
                self.figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}_{this_title}",
                    myformat="png",
                    subfolderpath=self.save_figure_to_folder,
                )

    def _get_df_for_xy_plot(self, mid, para, ana, ave, comp, comp_type, coll_ana_df):

        # Build path
        col_suffix = ""
        path_col_list = []
        col_dict = {}
        for this_mid in mid:
            for this_para in para:

                folder_name = f"{this_mid}_{this_para}"
                if comp is True:
                    folder_name = f"{folder_name}_compiled_results"
                    col_suffix = f"_{comp_type}"
                _path = self.context.path.joinpath(folder_name)

                # Name column
                for this_ana in ana:
                    try:
                        col_stem = coll_ana_df.loc[this_ana]["csv_col_name"]
                    except KeyError:
                        raise KeyError(f"to_mpa_dict missing key {this_ana}")

                    if (
                        col_stem[: col_stem.find("_")].lower() in ["meanfr", "edist"]
                        and col_suffix == "_accuracy"
                    ):
                        print(
                            f"{col_suffix} does not exist for 'meanfr' or 'edist', replacing with '_mean' "
                        )
                        col_suffix = "_mean"
                    col = f"{col_stem}{col_suffix}"
                    path_col_list.append([_path, col, col_suffix])
                    col_dict[this_ana] = col

        # Get df. This is the same w/ and wo/ ave
        df_coll = pd.DataFrame()
        for [this_path, this_col, col_suffix] in path_col_list:
            (
                data0_df,
                data_df_compiled,
                independent_var_col_list,
                dependent_var_col_list,
                time_stamp,
            ) = self.data_io.get_csv_as_df(
                folder_name=this_path, include_only=col_suffix
            )
            df_coll[this_col] = data_df_compiled[this_col]

        if ave is True:
            # Weighted average over NGs. Connections provide true N units.
            conn_data = self.data_io.get_data(
                Path.joinpath(self.context.input_folder, self.context.conn_file_out)
            )
            unit_positions_dict = conn_data["positions_all"]["z_coord"]
            full_data_ng_list = list(conn_data["positions_all"]["z_coord"].keys())

            valid_data_ng_dict = {
                col_dict[req_ng]: len(unit_positions_dict[ng])
                for ng in full_data_ng_list
                for req_ng in col_dict.keys()
                if ng in col_dict[req_ng]
            }

            print(f"\nAveraging with {valid_data_ng_dict} weights...")

            # Generate new col with weighted averages
            w_ave_col_name = "w_ave"
            data_df_compiled[w_ave_col_name] = 0

            for this_key in col_dict.keys():
                this_column = col_dict[this_key]
                data_df_compiled[w_ave_col_name] = data_df_compiled[w_ave_col_name] + (
                    data_df_compiled[this_column] * valid_data_ng_dict[this_column]
                ) / sum(valid_data_ng_dict.values())
            data_columns_list = [w_ave_col_name]

        elif ave is False:
            data_columns_list = list(col_dict.values())

        return data_df_compiled[data_columns_list]

    def _get_df_for_binned_lineplot(
        self, mid, para, ana, ave, comp, comp_type, coll_ana_df, x_data=None
    ):

        # Build path
        col_suffix = ""
        path_col_list = []
        col_dict = {}
        for this_mid in mid:
            for this_para in para:

                folder_name = f"{this_mid}_{this_para}"
                if comp is True:
                    folder_name = f"{folder_name}_compiled_results"
                    col_suffix = f"_{comp_type}"
                _path = self.context.path.joinpath(folder_name)

                # Name column
                for this_ana in ana:
                    try:
                        col_stem = coll_ana_df.loc[this_ana]["csv_col_name"]
                    except KeyError:
                        raise KeyError(f"to_mpa_dict missing key {this_ana}")

                    if (
                        col_stem[: col_stem.find("_")].lower() in ["meanfr", "edist"]
                        and col_suffix == "_accuracy"
                    ):
                        print(
                            f"{col_suffix} does not exist for 'meanfr' or 'edist', replacing with '_mean' "
                        )
                        col_suffix = "_mean"
                    col = f"{col_stem}{col_suffix}"
                    path_col_list.append([_path, col, col_suffix, this_mid, this_para])
                    col_dict[this_ana] = col

        # Get df. This is the same w/ and wo/ ave
        # get columns
        cols = ["Analysis", "Values", "Midpoint", "Parameter"]

        df_coll = pd.DataFrame(columns=cols)
        for [this_path, this_col, col_suffix, this_mid, this_para] in path_col_list:
            (
                data0_df,
                data_df_compiled,
                independent_var_col_list,
                dependent_var_col_list,
                time_stamp,
            ) = self.data_io.get_csv_as_df(
                folder_name=this_path, include_only=col_suffix
            )

            data_df_tmp = pd.DataFrame(columns=cols)
            data_df_tmp["Values"] = data_df_compiled[this_col]
            data_df_tmp["Analysis"] = this_col
            data_df_tmp["Midpoint"] = this_mid
            data_df_tmp["Parameter"] = this_para
            if x_data is not None:
                data_df_tmp2 = x_data[
                    (x_data["Midpoint"] == this_mid)
                    & (x_data["Parameter"] == this_para)
                ]
                data_df_tmp2.reset_index(inplace=True)
                x_cols_names = list(set(data_df_tmp2["Analysis"].values))
                if len(set(x_cols_names)) > 1:  # If more the one type of x
                    raise NotImplementedError(
                        "More than one x analysis not yet implemented for kind binned_lineplot, aborting..."
                    )
                else:
                    data_df_tmp[x_cols_names[0]] = data_df_tmp2["Values_binned"]
            df_coll = pd.concat([df_coll, data_df_tmp], axis=0, ignore_index=True)

        return df_coll, col_dict

    def rename_duplicate_columns(self, x_df, y_df, data_df_compiled):
        """
        Check if x_df and y_df have any same column names
        If yes, rename them to avoid duplicate column names
        """
        cols_x = list(x_df.columns)
        cols_y = list(y_df.columns)
        cols_duplicate = list(set(cols_x).intersection(cols_y))
        if len(cols_duplicate) > 0:
            for col in cols_duplicate:
                x_df.rename(columns={col: f"{col}_x"}, inplace=True)
                y_df.rename(columns={col: f"{col}_y"}, inplace=True)

        data_df_compiled = pd.concat([x_df, y_df], axis=1)

        return x_df, y_df, data_df_compiled

    def show_xy_plot(
        self,
        xy_plot_dict,
    ):

        """ """

        coll_ana_df = self.coll_mpa_dict["coll_ana_df"]

        # Confirm list format
        keys_to_check = ["x_ana", "x_mid", "x_para", "y_ana", "y_mid", "y_para"]
        for this_key in keys_to_check:
            if isinstance(xy_plot_dict[this_key], str):
                xy_plot_dict[this_key] = [xy_plot_dict[this_key]]

        kind = xy_plot_dict["kind"]

        # Wide format df, only single mid and param, allows averaging of neuron groups:s (e.g. firing rates)
        if kind == "regplot":
            assert np.all(
                [
                    len(xy_plot_dict["x_mid"]) == 1,
                    len(xy_plot_dict["x_para"]) == 1,
                    len(xy_plot_dict["y_mid"]) == 1,
                    len(xy_plot_dict["y_para"]) == 1,
                ]
            ), "regplot accepts only single midpoints and parameters at a time, aborting..."

            x_df = self._get_df_for_xy_plot(
                xy_plot_dict["x_mid"],
                xy_plot_dict["x_para"],
                xy_plot_dict["x_ana"],
                xy_plot_dict["x_ave"],
                xy_plot_dict["compiled_results"],
                xy_plot_dict["compiled_type"],
                coll_ana_df,
            )

            y_df = self._get_df_for_xy_plot(
                xy_plot_dict["y_mid"],
                xy_plot_dict["y_para"],
                xy_plot_dict["y_ana"],
                xy_plot_dict["y_ave"],
                xy_plot_dict["compiled_results"],
                xy_plot_dict["compiled_type"],
                coll_ana_df,
            )

            data_df_compiled = pd.concat([x_df, y_df], axis=1)  #
            x_df, y_df, data_df_compiled = self.rename_duplicate_columns(
                x_df, y_df, data_df_compiled
            )

        # Long format df, accepts multiple mids and params, no averaging
        elif kind == "binned_lineplot":
            x_df, x_cols = self._get_df_for_binned_lineplot(
                xy_plot_dict["x_mid"],
                xy_plot_dict["x_para"],
                xy_plot_dict["x_ana"],
                xy_plot_dict["x_ave"],
                xy_plot_dict["compiled_results"],
                xy_plot_dict["compiled_type"],
                coll_ana_df,
            )

            n_bins = xy_plot_dict["n_bins"]
            x_max = x_df["Values"].max()
            x_min = x_df["Values"].min()
            bins = np.floor(np.linspace(x_min, x_max, n_bins))
            x_values_binned = np.digitize(x_df["Values"].values, bins)
            x_df["Values_binned"] = x_values_binned
            x_bin_idx = np.arange(len(bins)) + 1
            x_df["Values_binned"] = x_df["Values_binned"].replace(x_bin_idx, bins)

            y_df, y_cols = self._get_df_for_binned_lineplot(
                xy_plot_dict["y_mid"],
                xy_plot_dict["y_para"],
                xy_plot_dict["y_ana"],
                xy_plot_dict["y_ave"],
                xy_plot_dict["compiled_results"],
                xy_plot_dict["compiled_type"],
                coll_ana_df,
                x_data=x_df,
            )

            data_df_compiled = y_df  #

        self.data_is_valid(x_df.values, accept_empty=False)
        self.data_is_valid(y_df.values, accept_empty=False)

        xlog = xy_plot_dict["xlog"]
        ylog = xy_plot_dict["ylog"]

        if kind == "regplot":
            fig, axs = plt.subplots(
                len(x_df.columns),
                len(y_df.columns),
                squeeze=False,
                sharey=xy_plot_dict["sharey"],
            )

            scatter_kws = {"linewidth": 0, "s": 10, "color": "black"}  # s is size

            for row_idx, this_x_column in enumerate(x_df.columns):
                for col_idx, this_y_column in enumerate(y_df.columns):
                    sns.regplot(
                        x=this_x_column,
                        y=this_y_column,
                        data=data_df_compiled,
                        fit_reg=xy_plot_dict["draw_regression"],
                        ax=axs[row_idx, col_idx],
                        scatter_kws=scatter_kws,
                        order=xy_plot_dict["order"],
                    )

                    """
                    Calculate goodness-of-fit, correlation coefficient, and p-value for the correlation coefficient of the x and y data. 
                    """
                    if xy_plot_dict["draw_regression"] is True:
                        r, p = self.ana.stat_tests.pearson_correlation(
                            data_df_compiled[this_x_column],
                            data_df_compiled[this_y_column],
                        )
                        # Add r, p as text to regplot
                        axs[row_idx, col_idx].text(
                            0.05,
                            0.95,
                            f"r={r:.2f}, p={p:.3f}",
                            transform=axs[row_idx, col_idx].transAxes,
                            fontsize=10,
                            verticalalignment="top",
                            bbox=dict(facecolor="white", alpha=0.5),
                        )

                    if xy_plot_dict["draw_diagonal"] == True:
                        x_min = x_df[this_x_column].min()
                        x_max = x_df[this_x_column].max()
                        axs[row_idx, col_idx].plot(
                            [x_min, x_max],
                            [x_min, x_max],
                            "k--",
                        )

        elif kind == "binned_lineplot":
            fig, axs = plt.subplots(
                len(xy_plot_dict["x_ana"]),
                len(xy_plot_dict["y_ana"]),
                squeeze=False,
                sharey=xy_plot_dict["sharey"],
            )

            for row_idx, this_x_name in enumerate(x_cols.values()):
                for col_idx, this_y_name in enumerate(y_cols.values()):

                    this_data = data_df_compiled[
                        data_df_compiled["Analysis"] == this_y_name
                    ]
                    sns.lineplot(
                        x=this_x_name,
                        y="Values",
                        data=this_data,
                        ax=axs[row_idx, col_idx],
                        hue=xy_plot_dict["hue"],
                        ci=95,
                        n_boot=1000,
                    )

                    this_title = coll_ana_df.index[
                        coll_ana_df["csv_col_name"]
                        == this_y_name.replace(f"_{xy_plot_dict['compiled_type']}", "")
                    ].values
                    if row_idx == 0:
                        axs[row_idx, col_idx].set_title(this_title[0])

        if xlog is True:
            axs[row_idx, col_idx].set_xscale("log")
        if ylog is True:
            axs[row_idx, col_idx].set_yscale("log")

        if self.save_figure_with_arrayidentifier is not None:

            # Assuming the last word in this_NG_name contains the necessary id
            id = (
                "_".join(y_df.columns)[:2] + "_vs_" + "_".join(x_df.columns)[:2] + "_XY"
            )

            self.figsave(
                figurename=f"{self.save_figure_with_arrayidentifier}_{id}",
                myformat="svg",
                subfolderpath=self.save_figure_to_folder,
            )
            self.figsave(
                figurename=f"{self.save_figure_with_arrayidentifier}_{id}",
                myformat="png",
                subfolderpath=self.save_figure_to_folder,
            )

    def show_IxO_conf_mtx(
        self,
        midpoint="",
        parameter="",
        ana_list=[],
        ana_suffix_list=[],
        par_value_string_list=[],
        best_is_list=[],
    ):
        """
        At the moment this works only for 2D data
        """
        # Get IxO...gz data from all midpoint_parameter_iterations
        assert all(
            [midpoint, parameter, ana_list, ana_suffix_list, best_is_list]
        ), "You need to define all input parameteres, sorry, aborting..."
        assert (
            len(ana_list) == len(ana_suffix_list) == len(best_is_list)
        ), "The parameter lists are not of equal length, aborting..."

        # Get a list of files
        iteration_paths_list = self.data_io.listdir_loop(
            path=self.context.path,
            data_type=f"{midpoint}_{parameter}_",
            exclude="_compiled_results",
        )
        IxO_file_list = []
        metadata_file_list = []
        for this_path in iteration_paths_list:
            metadata_file_list_tmp = self.data_io.listdir_loop(
                path=this_path, data_type="metadata_", exclude=None
            )
            if len(metadata_file_list_tmp) > 1:
                # Assuming updated metadata files
                metadata_file_list += [
                    m for m in metadata_file_list_tmp if "_updated." in str(m)
                ]
            else:
                metadata_file_list += metadata_file_list_tmp
            IxO_file_list += self.data_io.listdir_loop(
                path=this_path, data_type="IxO_analysis", exclude=None
            )

        assert len(IxO_file_list) == len(
            metadata_file_list
        ), "Unequal N metadata (iter runs) and IxO files (iter analysis), check files, aborting..."

        IxO_file_list = sorted(IxO_file_list)
        metadata_file_list = sorted(metadata_file_list)
        # Load IxO data into one array.
        IxO_mtx_0 = self.data_io.get_data(IxO_file_list[0])
        data_dims = IxO_mtx_0["data_dims"]
        n_iterations = len(IxO_file_list)

        full_data_mtx_nan = (
            np.zeros([data_dims[0], data_dims[1], data_dims[2], n_iterations]) * np.nan
        )

        fig, axs = plt.subplots(1, len(ana_list), squeeze=False)

        for this_analysis_idx, (
            this_analysis,
            this_analysis_substring,
            best_is,
        ) in enumerate(zip(ana_list, ana_suffix_list, best_is_list)):

            full_data_mtx = full_data_mtx_nan.copy()

            for this_idx, this_file in enumerate(IxO_file_list):
                IxO_data_dict = self.data_io.get_data(this_file)
                if this_idx == 0:
                    keys_list = IxO_data_dict.keys()
                    analysis_keys_all = [
                        ak
                        for ak in keys_list
                        if ak.lower().startswith(this_analysis.lower())
                    ]
                    analysis_key = [
                        ak
                        for ak in analysis_keys_all
                        if this_analysis_substring.lower() in ak.lower()
                    ]

                    assert (
                        len(analysis_key) == 1
                    ), "There should be exactly one ana_list key for show IxO confusion mtx, aborting..."
                full_data_mtx[:, :, :, this_idx] = IxO_data_dict[analysis_key[0]]

            # Get iterations of parameter values in one array
            # Get metadata df to find parameter match in full data mtx. This is our index to full data mtx 3rd dim
            data_df0 = self.data_io.get_data(metadata_file_list[0])
            mask = (data_df0["Dimension-1 Value"] == par_value_string_list[0]) & (
                data_df0["Dimension-2 Value"] == par_value_string_list[1]
            )
            idx = data_df0[mask].index[0]

            # Select ana_list parameter I_x E_y values x iterations x 3 x 3
            data_mtx_np = full_data_mtx[:, :, idx, :]

            # Find best max/min values row-wise
            if best_is == "max":
                if "TransfEntropy" in this_analysis_substring:
                    data_mtx_np = np.nan_to_num(data_mtx_np, copy=True, nan=0.0)
                this_ixo_bool = (
                    data_mtx_np == data_mtx_np.max(axis=1)[:, None]
                ).astype(int)
            elif best_is == "min":
                this_ixo_bool = (
                    data_mtx_np == data_mtx_np.min(axis=1)[:, None]
                ).astype(int)
            # Sum across iterations (dim 2)
            this_ixo_sum = this_ixo_bool.sum(axis=2)

            # show conf mtx
            sns.heatmap(
                this_ixo_sum,
                cmap=self.cmap,
                ax=axs[0][this_analysis_idx],
                cbar=True,
                annot=True,
                fmt=".2g",
                vmin=0,
                vmax=this_ixo_bool.shape[2],  # N iterations
                linewidths=0.5,
                linecolor="black",
            )
            axs[0][this_analysis_idx].set_title(this_analysis)

            if self.save_figure_with_arrayidentifier is not None:

                id = "conf_mtx"

                self.figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}",
                    myformat="svg",
                    subfolderpath=self.save_figure_to_folder,
                )
                self.figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}",
                    myformat="png",
                    subfolderpath=self.save_figure_to_folder,
                )

    def show_optimal_value_analysis(self, data_for_viz, savefigname=None):
        """
        Visualize the optimal value analysis.
        """
        data, analyze, figure_type = data_for_viz
        if figure_type == "full_mtx":
            # Plot the full matrix as seaborn heatmap
            plt.figure()
            ax = sns.heatmap(
                data,
                cmap="gist_earth",
                cbar=True,
                annot=True,
                fmt=".2g",
            )
            plt.title(f"{analyze} matrix")
            ax.set_xlabel("Target")
            ax.set_ylabel("Source")

            if savefigname:
                self.figsave(
                    figurename=savefigname,
                    myformat="png",
                    subfolderpath="Analysis_Figures",
                )
                self.figsave(
                    figurename=savefigname,
                    myformat="svg",
                    subfolderpath="Analysis_Figures",
                )

        elif figure_type == "value_vs_delay":
            delay_array_in_milliseconds, value_array, noise_array, delay_report = data
            plt.plot(delay_array_in_milliseconds, value_array, label="Value")
            plt.plot(delay_array_in_milliseconds, noise_array, label="Noise")
            plt.title(analyze, fontsize=12)
            plt.text(
                0.9,
                0.2,
                delay_report,
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=8,
                transform=plt.gcf().transFigure,
            )

            plt.xlabel("Delay (ms)")
            plt.ylabel("Value")

            if savefigname:
                self.figsave(
                    figurename=savefigname,
                    myformat="png",
                    subfolderpath="Analysis_Figures",
                )
                self.figsave(
                    figurename=savefigname,
                    myformat="svg",
                    subfolderpath="Analysis_Figures",
                )

    def show_iter_optimal_value_analysis(self, folderpath, savefigname=None):
        """
        After running optimal value analyzes for iterations, show the results.
        """

        # Get csv filenames from folderpath.
        csv_file_list = self.data_io.listdir_loop(
            folderpath, data_type=".csv", exclude=None
        )
        [optimal_full_path] = [
            f for f in csv_file_list if str(f.name).startswith("optimal_values_df")
        ]
        [nonoptimal_full_path] = [
            f for f in csv_file_list if str(f.name).startswith("nonoptimal_values_df")
        ]

        # Load both csv files to optimal and nonoptimal df:s from the folderpath
        # Use pandas read from csv function
        optimal_df = pd.read_csv(optimal_full_path)
        nonoptimal_df = pd.read_csv(nonoptimal_full_path)

        analyzes_list = self.context.to_mpa_dict["analyzes"]
        analyzes_col_name_list = []
        for this_ana in analyzes_list:
            analyzes_col_name_list.append(
                self.coll_mpa_dict["coll_ana_df"].loc[this_ana]["csv_col_name"]
            )

        # Create figure with one row of subplots. The N subplots equals optimal_df columns with suffix "_mean"
        fig, axs = plt.subplots(1, len(analyzes_col_name_list), sharey=False)

        # Loop over all analyzes
        for this_col_idx, this_col_name in enumerate(analyzes_col_name_list):
            # Get the optimal value for this analysis
            optimal_value = optimal_df[this_col_name + "_mean"].values
            # Get the nonoptimal value for this analysis
            nonoptimal_value = nonoptimal_df[this_col_name + "_mean"].values
            # Get the optimal value for this analysis
            optimal_value_std = optimal_df[this_col_name + "_SD"].values
            # Get the nonoptimal value for this analysis
            nonoptimal_value_std = nonoptimal_df[this_col_name + "_SD"].values
            # Get the delay_in_ms for this analysis
            delay_in_ms = optimal_df["delay_in_ms"].values

            # Make a new long-format dataframe for plotting.
            # The columns are: delay_in_ms, optimal_value, nonoptimal_value, optimal_value_std, nonoptimal_value_std
            df_for_plotting = pd.DataFrame(
                {
                    "delay_in_ms": delay_in_ms,
                    "optimal_value": optimal_value,
                    "nonoptimal_value": nonoptimal_value,
                    "optimal_value_std": optimal_value_std,
                    "nonoptimal_value_std": nonoptimal_value_std,
                }
            )

            # Plot the dataframe
            sns.lineplot(
                x="delay_in_ms",
                y="optimal_value",
                data=df_for_plotting,
                ax=axs[this_col_idx],
                label="Optimal",
                color="blue",
                linewidth=0.5,
            )
            sns.lineplot(
                x="delay_in_ms",
                y="nonoptimal_value",
                data=df_for_plotting,
                ax=axs[this_col_idx],
                label="Nonoptimal",
                color="red",
                linewidth=0.5,
            )

            # Plot SD for optimal and nonoptimal
            lower_bound_optimal = (
                df_for_plotting["optimal_value"] - df_for_plotting["optimal_value_std"]
            )
            upper_bound_optimal = (
                df_for_plotting["optimal_value"] + df_for_plotting["optimal_value_std"]
            )
            axs[this_col_idx].fill_between(
                df_for_plotting["delay_in_ms"],
                y1=lower_bound_optimal,
                y2=upper_bound_optimal,
                color="blue",
                alpha=0.3,
            )

            lower_bound_nonoptimal = (
                df_for_plotting["nonoptimal_value"]
                - df_for_plotting["nonoptimal_value_std"]
            )
            upper_bound_nonoptimal = (
                df_for_plotting["nonoptimal_value"]
                + df_for_plotting["nonoptimal_value_std"]
            )
            axs[this_col_idx].fill_between(
                df_for_plotting["delay_in_ms"],
                y1=lower_bound_nonoptimal,
                y2=upper_bound_nonoptimal,
                color="red",
                alpha=0.3,
            )

            axs[this_col_idx].set_title(analyzes_list[this_col_idx])
            axs[this_col_idx].set_xlabel("Delay (ms)")
            axs[this_col_idx].set_ylabel("Value")
            axs[this_col_idx].legend()

        if savefigname:
            self.figsave(
                figurename=savefigname, myformat="png", subfolderpath="Analysis_Figures"
            )
            self.figsave(
                figurename=savefigname, myformat="svg", subfolderpath="Analysis_Figures"
            )


if __name__ == "__main__":

    pass
