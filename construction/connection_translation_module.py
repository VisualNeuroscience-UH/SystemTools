import numpy as np

# Visualization
from matplotlib import pyplot as plt

# Builtins
from pathlib import Path
import pdb


class ConnectionTranslator:
    # This is what this class needs from context.
    # After construction, these attributes are under
    # self.context
    _properties_list = [
        "path",
        "matlab_workspace_file",
        "conn_skeleton_file_in",
        "conn_file_out",
        "input_filename",
        "input_folder",
    ]

    def __init__(self, context, data_io) -> None:

        self.context = context.set_context(self._properties_list)
        self.data_io = data_io

    def _show_histogram(
        self, data, figure_title=None, skip_under_one_pros=False, bins=10
    ):

        if "scipy.sparse.csr.csr_matrix" in str(type(data)):
            data = data.toarray()

        if "numpy.ndarray" in str(type(data)):
            data = data.flatten()

        if skip_under_one_pros == True:
            # data = data[data != 0]
            one_pros = np.max(data) / 100
            data = data[data > one_pros]

        if figure_title == None:
            figure_title = ""
        fig = plt.figure()
        plt.hist(data, bins=bins)
        fig.suptitle(f"{figure_title}", fontsize=12)

    def replace_conn(
        self,
        show_histograms=False,
        constant_scaling=False,
        constant_value=1e-9,
        randomize_conn_list=[],
    ):
        """
        After creating a CxSystem neural system with correct cell numbers and random connectivity, here we assign
        precomputed connection weights to this system.
        """
        mat_data_dict = self.data_io.get_data(self.context.matlab_workspace_file)
        conn_skeleton_dict = self.data_io.get_data(self.context.conn_skeleton_file_in)

        mat_keys = ["FsE", "CsEE", "CsEI", "CsIE", "CsII", "DecsE", "DecsI"]
        mat_data_dict_keys_str = str(mat_data_dict.keys())

        assert all(
            [x in mat_data_dict_keys_str for x in mat_keys]
        ), "Some mat keys not found, aborting..."

        match_conn_names = {
            "relay_vpm__to__L4_CI_SS_L4_soma": "FsE",
            "L4_CI_SS_L4__to__L4_CI_SS_L4_soma": "CsEE",
            "L4_CI_SS_L4__to__L4_CI_BC_L4_soma": "CsEI",
            "L4_CI_BC_L4__to__L4_CI_SS_L4_soma": "CsIE",
            "L4_CI_BC_L4__to__L4_CI_BC_L4_soma": "CsII",
            "L4_CI_SS_L4__to__L4_SS_L4_soma": "DecsE",
            "L4_CI_BC_L4__to__L4_SS_L4_soma": "DecsI",
        }

        # which phase of learned connections to select. 29 = after teaching
        mat_teach_idx = 28
        conn_final_dict = conn_skeleton_dict

        # We need to turn negative inhibitory connections in matlab to positive for CxSystem.
        # These connections are fed to gi which has no driving force, because they are I_NDF type.
        # There the conductance itself is negative, which is necessary if we want inhibition without
        # driving force. The DecsI has both negative and positive connection strengths
        # (optimized for decoding in the orginal code).
        inh_keys = ["CsIE", "CsII", "DecsI"]

        for this_conn in match_conn_names.keys():
            # Get cxsystem connection strengths (i.e. Physiology parameters J, J_I, k*J or k_I*J_I
            # multiplied by n synapses/connection)
            data_cx = conn_skeleton_dict[this_conn]["data"]
            # Get mat_data connection strengths. Transpose because unintuitively (post,pre), except for FsE
            data_mat = mat_data_dict[match_conn_names[this_conn]][mat_teach_idx, :, :].T
            # FsE is the only (pre,post) in matlab code (SIC!)
            if match_conn_names[this_conn] == "FsE":
                data_mat = data_mat.T

            assert (
                data_mat.shape == data_cx.shape
            ), "Connection shape mismatch, aborting..."

            # Scale mat_data to min and max values of cxsystem connection strengths (excluding zeros)
            # In constant scaling, just scale with constant_value without any other transformations
            if constant_scaling is False:
                data_out = self.scale_values(
                    data_mat, target_data=data_cx, skip_under_zeros_in_scaling=False
                )
            elif constant_scaling is True:
                data_out = self.scale_with_constant(
                    data_mat, constant_value=constant_value
                )

            # Turn negative inhibitory connections in matlab to positive for CxSystem
            if match_conn_names[this_conn] in inh_keys:
                data_out = data_out * -1

            # Randomize by request for control conditions
            if match_conn_names[this_conn] in randomize_conn_list:
                rng = np.random.default_rng()
                # Randomly permute post connections separately for each pre unit.
                data_out = rng.permuted(data_out, axis=1)

            # viz by request
            if show_histograms is True:
                self._show_histogram(
                    data_cx, figure_title=this_conn, skip_under_one_pros=False
                )
                self._show_histogram(
                    data_mat,
                    figure_title=match_conn_names[this_conn],
                    skip_under_one_pros=False,
                )
                # L4_BC_L4__to__L4_CI_SS_L4_soma_out
                self._show_histogram(
                    data_out,
                    figure_title=this_conn + "_out",
                    skip_under_one_pros=False,
                )
                plt.show()

            # return scaled values
            conn_final_dict[this_conn]["data"] = self.data_io.csr_matrix(data_out)
            conn_final_dict[this_conn]["n"] = 1

        savepath = Path.joinpath(self.context.input_folder, self.context.conn_file_out)
        self.data_io.write_to_file(savepath, conn_final_dict)

    def scale_with_constant(self, source_data, constant_value=1e-9):
        """
        Scale data with constant value. 1e-9 scales to nano-scale, corresponding to nS-level conductances
        """

        data_out = source_data * constant_value

        return data_out

    def scale_values(
        self,
        source_data,
        target_data=None,
        skip_under_zeros_in_scaling=True,
        preserve_sign=True,
    ):
        """
        Scale data to same distribution but between target data min and max values.
        If target data are missing, normalizes to 0-1
        -skip_under_zeros_in_scaling: When True, assumes values at zero or negative are minor and not
        contributing to responses.
        -preserve sign: let negative weights remain negative
        """
        if target_data is None:
            min_value = 0
            max_value = 1
        else:
            if skip_under_zeros_in_scaling is True:
                target_data = target_data[target_data >= 0]
            min_value = np.min(target_data)
            max_value = np.max(target_data)

        if skip_under_zeros_in_scaling is True:
            source_data_nonzero_idx = source_data >= 0
            source_data_shape = source_data.shape
            data_out = np.zeros(source_data_shape)
            source_data = source_data[source_data_nonzero_idx]
            print(
                f"Scaling {(source_data.size / data_out.size) * 100:.0f} percent of data, rest is considered zero"
            )

        # Shift to zero
        shift_distance = np.min(source_data)
        data_minzero = source_data - shift_distance
        # Scale to destination range
        scaling_factor = np.ptp([min_value, max_value]) / np.ptp(
            [np.min(source_data), np.max(source_data)]
        )
        data_scaled_minzero = data_minzero * scaling_factor

        # Add destination min
        data_scaled = data_scaled_minzero + min_value

        if preserve_sign is True and shift_distance < 0:
            data_scaled = data_scaled + (shift_distance * scaling_factor)

        if skip_under_zeros_in_scaling is True:
            data_out[source_data_nonzero_idx] = data_scaled
        else:
            data_out = data_scaled

        return data_out

    def create_current_injection(self, randomize=False):
        # Multiply time x Nx matrix of Input with Nx x Nunits matrix of original FE mapping
        # (input to excitatory neuron group mapping). Ref Brendel_2020_PLoSComputNeurosci

        # Get FE -- from input forward to e group connection matrix
        mat_data_dict = self.data_io.get_data(self.context.matlab_workspace_file)
        mat_key = "FsE"
        FE_all_learning_steps = mat_data_dict[mat_key]
        # Last learning step FE shape (3, 300)
        FE = FE_all_learning_steps[-1, :, :]
        if randomize is True:
            rng = np.random.default_rng()
            # Randomly permute post connections separately for each pre unit.
            FE = rng.permuted(FE, axis=1)

        # Read existing Input
        input_filename = self.context.input_filename
        assert input_filename is not None, "Input mat filename not set, aborting..."
        assert (
            input_filename.as_posix()[-4:] == ".mat"
        ), "Input filename does not end .mat, aborting..."
        input_filename_full = Path.joinpath(self.context.input_folder, input_filename)
        input_dict = self.data_io.get_data(input_filename_full)

        # Extract Input
        Input = input_dict["stimulus"]  # Input.shape (3, 10000)

        # Multiply Input x FE to get injected current
        injected_current = np.dot(Input.T, FE)

        # Save mat file for the current injection
        if randomize is True:
            current_injection_filename_full = Path(
                input_filename_full.as_posix()[:-4] + "_permuted_ci.mat"
            )
        else:
            current_injection_filename_full = Path(
                input_filename_full.as_posix()[:-4] + "_ci.mat"
            )
        # This will be read by physiology_reference.py
        mat_out_dict = {
            "injected_current": injected_current,
            "dt": input_dict["frameduration"],
            "stimulus_duration_in_seconds": input_dict["stimulus_duration_in_seconds"],
        }
        self.data_io.savemat(current_injection_filename_full, mat_out_dict)

    def create_control_conn(self, conn="ALL"):
        # Randomize learned connectivity for control conditions.
        # 'EI' all mutual NG1 and NG2 connections
        # 'D' from EI to output
        # 'ALL' all of the above

        if conn == "EI":
            randomize_conn_list = ["CsEE", "CsEI", "CsIE", "CsII"]
        elif conn == "D":
            randomize_conn_list = ["DecsE", "DecsI"]
        elif conn == "ALL":
            randomize_conn_list = ["CsEE", "CsEI", "CsIE", "CsII", "DecsE", "DecsI"]
        else:
            raise TypeError("Unknown connections keyword, aborting...")

        # Update self.context.conn_file_out. It will be used inside replace_conn.

        filename = self.context.conn_file_out.stem
        file_extension = self.context.conn_file_out.suffix

        self.context.conn_file_out = Path(
            filename + "_permuted_" + conn + file_extension
        )

        self.replace_conn(
            show_histograms=False,
            constant_scaling=True,
            constant_value=1e-9,
            randomize_conn_list=randomize_conn_list,
        )
