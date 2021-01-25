# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.io as sio

# Computational neuroscience
import brian2.units as b2u
from cxsystem2.core.tools import write_to_file

# Builtin
import os
import sys

# Current repo
from system_analysis import SystemAnalysis

# Develop
import pdb

'''
Module on project-specific data and analysis.

Prepares data for system_analysis. These data are fed to system_analysis init, where they are processes 
and visualized.

PSEUDOCODE:
get_connections
path in
list Fig4 mat variables

connectivities (dimensions):
FE (3,300) = input group to E = IN2E NOTE that this is transposed compared to others. This is (pre,post)
CsEE (29, 300, 300) = E2E (the first 29 is development at log intervals of the 300:300 (post,pre)conn matrix)
CsEI (29, 75, 300) = E2I (post,pre)
CsIE (29, 300, 75) = I2E (post,pre)
CsII (29, 75, 75) =I2I (post,pre)

Optimal decoders for each instance of the registered connectivities.
DecsE (29, 3, 300) = E2D (post,pre)
DecI  (29, 3, 75) = I2D (post,pre)

read Fig4 mat files. 

Learn the CxSystem conn data type 

turn conn.mat to conn.gz


path out
save

Developed by Simo Vanni 2021
'''

class Project(SystemAnalysis):

    def __init__(self, path='./', experiment_folder=None, mat_filename=None, connection_filename_out=None, 
                connection_skeleton_filename_in=None):

        self.path = path
        self.experiment_folder = experiment_folder
        self.mat_filename = mat_filename
        self.connection_filename_out = connection_filename_out
        self.connection_skeleton_filename_in = connection_skeleton_filename_in

    def _show_histogram(self, data, figure_title=None, skip_under_one_pros=False, bins=10):

        if 'scipy.sparse.csr.csr_matrix' in str(type(data)):
            data = data.toarray()
        
        if 'numpy.ndarray' in str(type(data)):
            data = data.flatten()

        if skip_under_one_pros == True:
            # data = data[data != 0]
            one_pros = np.max(data) / 100
            data = data[data > one_pros]

        if figure_title == None:
            figure_title = ''

        fig = plt.figure()
        plt.hist(data, bins=bins)
        fig.suptitle(f'{figure_title}', fontsize=12)
    
    def scale_values(self, source_data, target_data = None, skip_under_zeros_in_scaling=True):
        '''
        Scale data to same distribution but between target data min and max values.
        If target data are missing, normalizes to 0-1
        -skip_under_zeros_in_scaling: When True, assumes values at zero or negative are minor and not 
        contributing to responses
        '''
        if target_data is None:
            min_value=0
            max_value=1
        else:
            if skip_under_zeros_in_scaling is True:
                target_data = target_data[target_data > 0]
            min_value = np.min(target_data)
            max_value = np.max(target_data)

        if skip_under_zeros_in_scaling is True:
            source_data_nonzero_idx = source_data>0
            source_data_shape = source_data.shape
            data_out = np.zeros(source_data_shape)
            source_data = source_data[source_data_nonzero_idx]
            print(f'Scaling {(source_data.size / data_out.size) * 100:.0f} percent of data, rest is considered zero')

        # Shift to zero
        data_minzero = (source_data - np.min(source_data))
        # Scale to destination range
        data_scaled_minzero = data_minzero * (np.ptp([min_value, max_value]) / np.max(source_data))
        # Add destination min
        data_scaled = data_scaled_minzero + min_value

        if skip_under_zeros_in_scaling is True:
            data_out[source_data_nonzero_idx] = data_scaled
        else:
            data_out = data_scaled

        return data_out 

    def replace_connections(self, show_histograms=False):

        mat_data_dict = self.getData(self.mat_filename)
        connection_skeleton_dict = self.getData(self.connection_skeleton_filename_in)

        mat_keys = ['FsE', 'CsEE', 'CsEI', 'CsIE', 'CsII', 'DecsE', 'DecsI']
        mat_data_dict_keys_str = str(mat_data_dict.keys())

        assert all([x in mat_data_dict_keys_str for x in mat_keys]), 'Some mat keys not found, aborting...'

        match_connection_names = {  'relay_video__to__L4_SS_L4_soma' : 'FsE',
                                    'L4_SS_L4__to__L4_SS_L4_soma' : 'CsEE',
                                    'L4_SS_L4__to__L4_BC_L4_soma' : 'CsEI',
                                    'L4_BC_L4__to__L4_SS_L4_soma' : 'CsIE',
                                    'L4_BC_L4__to__L4_BC_L4_soma' : 'CsII',
                                    'L4_SS_L4__to__L4_SS2_L4_soma' : 'DecsE',
                                    'L4_BC_L4__to__L4_SS2_L4_soma' : 'DecsI'
                                    }

        # which phase of learned connections to select. 29 = after teaching
        mat_teach_idx = 28 
        connection_final_dict = connection_skeleton_dict

        for this_connection in match_connection_names.keys():
            # Get cxsystem connection strengths (i.e. Physiology parameters J, J_I, k*J or k_I*J_I 
            # multiplied by n synapses/connection)
            data_cx = connection_skeleton_dict[this_connection]['data']
            # Get mat_data connection strengths. Transpose because unintuitively (post,pre), except for FsE
            data_mat = mat_data_dict[match_connection_names[this_connection]][mat_teach_idx,:,:].T
            if match_connection_names[this_connection] == 'FsE': # FsE is the only (pre,post) in matlab code (SIC!)
                data_mat = data_mat.T

            # DEBUG ONLY
            todays_interest = 'CsIE'
            connection_multiplier = 1 
            inh_conn = ['CsIE', 'CsII']
            mat_conn = match_connection_names[this_connection]
            if mat_conn == todays_interest:
                if mat_conn in str(inh_conn):
                    connection_multiplier = -1
                self._show_histogram(   data_mat * connection_multiplier, 
                                        figure_title=mat_conn + f' connection_multiplier = {connection_multiplier}', 
                                        skip_under_one_pros=True,
                                        bins = 100)
                plt.show()

            assert data_mat.shape == data_cx.shape, 'Connection shape mismatch, aborting...'

            # Scale mat_data to min and max values of cxsystem connection strengths (excluding zeros)
            data_out = self.scale_values(data_mat, target_data=data_cx, skip_under_zeros_in_scaling=False)

            # viz by request
            if show_histograms is True:
                self._show_histogram(data_cx, figure_title=this_connection, skip_under_zeros=True)
                self._show_histogram(data_mat, figure_title=match_connection_names[this_connection], skip_under_zeros=False)
                self._show_histogram(data_out, figure_title=this_connection + '_out', skip_under_zeros=False)

            # return scaled values
            connection_final_dict[this_connection] = data_out
            plt.show()

    def create_current_injection(self, input_filename=None):   
        # Multiply time x Nx matrix of Input with Nx x Nunits matrix of Deneve's FE mapping 
        # (input to excitatory neuron group mapping). Ref Brendel_2020_PLoSComputNeurosci

        # Get FE -- from input forward to e group connection matrix
        mat_data_dict = self.getData(self.mat_filename)
        mat_key = 'FsE'
        FE_all_learning_steps = mat_data_dict[mat_key]
        FE= FE_all_learning_steps[-1,:,:] # Last learning step FE shape (3, 300)

        # Read existing Input
        assert input_filename is not None, 'Input mat filename not set, aborting...'
        assert input_filename[-4:] == '.mat', 'Input filename does not end .mat, aborting...'
        input_filename_full = os.path.join(self.path, self.experiment_folder, input_filename)
        input_dict = self.getData(input_filename_full)

        # Extract Input
        Input = input_dict['stimulus'] # Input.shape (3, 10000)

        # Multiply Input x FE to get injected current
        injected_current = np.dot(Input.T, FE)

        # Save mat file for the current injection
        current_injection_filename_full = input_filename_full[:-4] + '_ci.mat'
        mat_out_dict = {'injected_current' : injected_current, 
                        'dt' : input_dict['frameduration'],
                        'stimulus_duration_in_seconds' : input_dict['stimulus_duration_in_seconds']}
        sio.savemat(current_injection_filename_full, mat_out_dict)

if __name__=='__main__':

    if sys.platform == 'linux':
        path = r'/opt/tomas/projects/Results/Deneve_param'
    elif sys.platform == 'win32':
        path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve'

    experiment_folder = 'Replica_test'
    mat_filename = 'Fig4_workspace.mat'
    connection_filename_out = 'connections_Fig4.gz'
    # connection_skeleton_filename_in = 'Replica_test_connections_20210118_1141594.gz'
    connection_skeleton_filename_in = 'Replica_test_connections_20210119_1436570.gz'

    P = Project(
        path=path, 
        experiment_folder=experiment_folder,
        mat_filename = mat_filename, 
        connection_filename_out = os.path.join(experiment_folder, connection_filename_out),
        connection_skeleton_filename_in = os.path.join(experiment_folder, connection_skeleton_filename_in))

    # input_filename = 'input_noise_210118.mat'
    # P.create_current_injection(input_filename = input_filename)

    # P.replace_connections()

    NG_name = 'NG3_L4_SS2_L4'
    # # filename = os.path.join(experiment_folder, 'Replica_test_results_20210119_1436570.gz')
    # filename = 'Replica_test_results_20210119_1436570.gz'
    # P.plot_readout_on_input(NG_name, filename=filename, normalize=False)
    P.show_spikes(filename=None, savefigname='')
    P.showVm(filename=None, savefigname='')

    plt.show()
