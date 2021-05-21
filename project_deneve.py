# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.io as sio
import scipy.sparse as scprs

# Computational neuroscience
import brian2.units as b2u
from cxsystem2.core.tools import write_to_file

# Builtin
import os
import sys

# Current repo
from system_viz import SystemViz

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

class Project(SystemViz):

    def __init__(self, path='./', input_folder=None, output_folder=None, mat_filename=None, 
                connection_skeleton_filename_in=None, connection_filename_out=None, input_filename=None, NG_name=None):

        self.path = path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.mat_filename = mat_filename
        self.connection_skeleton_filename_in = connection_skeleton_filename_in
        self.connection_filename_out = os.path.join(input_folder, connection_filename_out)
        self.input_filename = input_filename
        self.NG_name = NG_name

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
    
    def scale_with_constant(self, source_data, constant_value=1e-9):
        '''
        Scale data with constant value. 1e-9 scales to nano-scale, corresponding to nS-level conductances
        '''
        
        data_out = source_data * constant_value

        return data_out    

    def scale_values(self, source_data, target_data = None, skip_under_zeros_in_scaling=True, preserve_sign=True):
        '''
        Scale data to same distribution but between target data min and max values.
        If target data are missing, normalizes to 0-1
        -skip_under_zeros_in_scaling: When True, assumes values at zero or negative are minor and not 
        contributing to responses. 
        -preserve sign: let negative weights remain negative
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
        shift_distance = np.min(source_data)
        data_minzero = (source_data - shift_distance)
        # Scale to destination range
        scaling_factor = (np.ptp([min_value, max_value]) / np.ptp([np.min(source_data), np.max(source_data)]))
        data_scaled_minzero = data_minzero * scaling_factor
        # Add destination min
        data_scaled = data_scaled_minzero + min_value

        if preserve_sign is True:
            data_scaled = data_scaled + (shift_distance * scaling_factor)

        if skip_under_zeros_in_scaling is True:
            data_out[source_data_nonzero_idx] = data_scaled
        else:
            data_out = data_scaled

        return data_out 

    def replace_connections(self, show_histograms=False, constant_scaling=False, constant_value=1e-9):
        '''
        After creating a CxSystem neural system with correct cell numbers and random connectivity, here we assign 
        precomputed connection weights to this system. 
        '''

        mat_data_dict = self.getData(self.mat_filename)
        connection_skeleton_dict = self.getData(self.connection_skeleton_filename_in)

        mat_keys = ['FsE', 'CsEE', 'CsEI', 'CsIE', 'CsII', 'DecsE', 'DecsI']
        mat_data_dict_keys_str = str(mat_data_dict.keys())

        assert all([x in mat_data_dict_keys_str for x in mat_keys]), 'Some mat keys not found, aborting...'

        match_connection_names = {  'relay_vpm__to__L4_CI_SS_L4_soma' : 'FsE',
                                    'L4_CI_SS_L4__to__L4_CI_SS_L4_soma' : 'CsEE',
                                    'L4_CI_SS_L4__to__L4_CI_BC_L4_soma' : 'CsEI',
                                    'L4_CI_BC_L4__to__L4_CI_SS_L4_soma' : 'CsIE',
                                    'L4_CI_BC_L4__to__L4_CI_BC_L4_soma' : 'CsII',
                                    'L4_CI_SS_L4__to__L4_CI_SS2_L4_soma' : 'DecsE',
                                    'L4_CI_BC_L4__to__L4_CI_SS2_L4_soma' : 'DecsI'
                                    }

        # which phase of learned connections to select. 29 = after teaching
        mat_teach_idx = 28 
        connection_final_dict = connection_skeleton_dict

        # We need to turn Deneve's negative inhibitory connections to positive for CxSystem
        # These connections are fed to gi which has no driving force, because they are I_NDF type. 
        # There the conductance itself is negative, which is necessary if we want inhibition 
        # without driving force. The DecsI has both negative and positive connection strengths
        # (optimized for decoding in Deneve's code). 
        inh_keys = ['CsIE', 'CsII', 'DecsI']

        for this_connection in match_connection_names.keys():
            # Get cxsystem connection strengths (i.e. Physiology parameters J, J_I, k*J or k_I*J_I 
            # multiplied by n synapses/connection)
            data_cx = connection_skeleton_dict[this_connection]['data']
            # Get mat_data connection strengths. Transpose because unintuitively (post,pre), except for FsE
            data_mat = mat_data_dict[match_connection_names[this_connection]][mat_teach_idx,:,:].T
            if match_connection_names[this_connection] == 'FsE': # FsE is the only (pre,post) in matlab code (SIC!)
                data_mat = data_mat.T

            assert data_mat.shape == data_cx.shape, 'Connection shape mismatch, aborting...'

            # Scale mat_data to min and max values of cxsystem connection strengths (excluding zeros)
            # In constant scaling, just scale with constant_value without any other transformations
            if constant_scaling is False:
                data_out = self.scale_values(data_mat, target_data=data_cx, skip_under_zeros_in_scaling=False)
            elif constant_scaling is True:
                data_out = self.scale_with_constant(data_mat, constant_value=constant_value)

            # Turn Deneve's negative inhibitory connections to positive for CxSystem
            if match_connection_names[this_connection] in inh_keys:
                data_out = data_out * -1

            # viz by request
            if show_histograms is True:
                self._show_histogram(data_cx, figure_title=this_connection, skip_under_one_pros=False)
                self._show_histogram(data_mat, figure_title=match_connection_names[this_connection], skip_under_one_pros=False)
                # L4_BC_L4__to__L4_CI_SS_L4_soma_out
                self._show_histogram(data_out, figure_title=this_connection + '_out', skip_under_one_pros=False)
                plt.show()

            # return scaled values
            connection_final_dict[this_connection]['data'] = scprs.csr_matrix(data_out)
            connection_final_dict[this_connection]['n'] = 1

        savepath = os.path.join(self.path, self.connection_filename_out)
        write_to_file(savepath, connection_final_dict)

    def create_current_injection(self):   
        # Multiply time x Nx matrix of Input with Nx x Nunits matrix of Deneve's FE mapping 
        # (input to excitatory neuron group mapping). Ref Brendel_2020_PLoSComputNeurosci

        input_filename = self.input_filename

        # Get FE -- from input forward to e group connection matrix
        mat_data_dict = self.getData(self.mat_filename)
        mat_key = 'FsE'
        FE_all_learning_steps = mat_data_dict[mat_key]
        FE= FE_all_learning_steps[-1,:,:] # Last learning step FE shape (3, 300)

        # Read existing Input
        assert input_filename is not None, 'Input mat filename not set, aborting...'
        assert input_filename[-4:] == '.mat', 'Input filename does not end .mat, aborting...'
        input_filename_full = os.path.join(self.path, self.input_folder, input_filename)
        input_dict = self.getData(input_filename_full)

        # Extract Input
        Input = input_dict['stimulus'] # Input.shape (3, 10000)

        # Multiply Input x FE to get injected current
        injected_current = np.dot(Input.T, FE)

        # Save mat file for the current injection
        current_injection_filename_full = input_filename_full[:-4] + '_ci.mat'
        # This will be read by physiology_reference.py
        mat_out_dict = {'injected_current' : injected_current, 
                        'dt' : input_dict['frameduration'],
                        'stimulus_duration_in_seconds' : input_dict['stimulus_duration_in_seconds']}
        sio.savemat(current_injection_filename_full, mat_out_dict)

    def read_input_matfile(self, filename=None, variable='stimulus'):
        print(filename)
        assert filename is not None, 'Filename not defined for read_input_matfile(), aborting...'
        
        analog_input = self.getData(filename, data_type=None)
        analog_signal = analog_input[variable].T
        assert analog_signal.ndim == 2, 'input is not a 2-dim vector, aborting...'
        # analog_timestep = analog_input['frameduration']

        return analog_signal


if __name__=='__main__':

    if sys.platform == 'linux':
        path = r'/opt/tomas/projects/Results/Deneve_param'
    elif sys.platform == 'win32':
        path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'
        # path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'
        os.chdir(path)

    # Experiment-specific file, folder and neuron group names. Do not use reserved words, such as "results"
    input_folder = 'Replica_in'
    output_folder = 'out2'
    mat_filename = 'Fig4_workspace.mat'
    connection_skeleton_filename_in = 'Replica_skeleton_connections_20210211_1453238.gz'
    connection_filename_out = 'connections_deneve_ci_constant.gz'
    input_filename = 'noise_210406.mat' # 'input_noise_210408.mat' # 'input_quadratic_three_units_2s.mat' # 'input_noise_210408.mat'
    NG_name_for_output = 'NG3_L4_CI_SS2_L4'

    P = Project(path=path, input_folder=input_folder, output_folder=output_folder, 
                mat_filename=mat_filename, connection_skeleton_filename_in=connection_skeleton_filename_in, 
                connection_filename_out=connection_filename_out, input_filename=input_filename, 
                NG_name=NG_name_for_output)

    # ############################
    # ### Create project files ###
    # ############################

    # # Creates file named input_filename but with _ci.mat suffix to experiment folder
    # P.create_current_injection() 
    
    # # Transforms Deneve's simulation connections from .mat file to CxSystem .gz fromat.
    # # Creates connection_filename_out to input folder
    # P.replace_connections(show_histograms=True, constant_scaling=True, constant_value=1e-9)


    # ############################
    # ###### Analysis & Viz ######
    # ############################

    # ## Print metadata ##
    # # Use "data_type" with no filename, if you want the most recent file
    # metadata_df = P.getData(filename=None, data_type='metadata') 
    # P.pp_df_full(metadata_df)
    
    # ## Readout on input ##
    # P.show_readout_on_input(results_filename=None, normalize=False, unit_idx_list=[0])
    # # Available simulation_engines: 'cxsystem' and 'matlab'. Matlab needs filename. Available readout_groups 'E' and 'I'. 
    # # Target output is always input_leak
    # P.show_estimate_on_input(results_filename=None, simulation_engine='cxsystem', readout_group='E', unit_idx_list=[0]) 
    # P.show_input_to_readout_coherence(results_filename='out_results_20210429_1825009_gL2nS_gL40nS.gz', savefigname='',signal_pair=[0,0])


    # ## Show spikes and vm ##q
    # P.show_spikes(results_filename=None, savefigname='')
    # neuron_index = {'NG1_L4_CI_SS_L4' : 150, 'NG2_L4_CI_BC_L4' : 37, 'NG3_L4_CI_SS2_L4' : 1}
    # P.show_analog_results(results_filename='out_results_20210429_1825009_gL2nS_gL40nS.gz', savefigname='',param_name='vm',startswith='NG', neuron_index=neuron_index) 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='vclamp',startswith='NG') 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='gclamp',startswith='NG') 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='v_lowpass1',startswith='NG') 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='ge_soma',startswith='NG') 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='v_homeo',startswith='NG') 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='w_minus',startswith='S') 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='wght',startswith='S') 
    # P.show_analog_results(results_filename=None, savefigname='',param_name='A_LTD_u',startswith='S') 

    ## Show E and I currents ##
    # neuron_index = None
    # neuron_index = {'NG1_L4_CI_SS_L4' : 150, 'NG2_L4_CI_BC_L4' : 37, 'NG3_L4_CI_SS2_L4' : 1}
    # P.show_currents(results_filename=None, savefigname='', neuron_index=neuron_index) 
    
    # ## Show connections ##
    # P.show_connections(connections_filename=None, hist_from='L4_CI_BC_L4__to__L4_CI_SS_L4_soma', savefigname='')

    ## Analyse and show arrayrun data ##
    # Available analyses: 'MeanFR', 'MeanVm', 'EICurrentDiff', 'GrCaus', 'Coherence', 'MeanError'
    extra_GrCaus_attributes = {
        'max_time_lag_seconds': 0.1,
        'downsampling_factor': 40,
        'test_timelag': False,
        'do_bonferroni_correction': True,
        'gc_significance_level': 0.001,
        'save_gc_fit_diagnostics': True,
        'show_gc_fit_diagnostics_figure': False}  
    extra_MeanError_attributes = {
        'decoding_method':'least_squares'} 
    
    # If this is active, the displayed array analysis figures are saved as arrayIdentifier_analysis_identifier.svg
    # at your path
    # P.save_figure_with_arrayidentifier = 'puppeli'

    # P.analyze_arrayrun(metadata_filename=None, analysis='MeanError', **extra_MeanError_attributes)
    # P.show_analyzed_arrayrun(csv_filename=None, analysis='MeanError', variable_unit='a.u.') # Empty NG_id_list for all groups

    # P.analyze_arrayrun(metadata_filename=None, analysis='MeanFR', t_idx_start=0, t_idx_end=-1)
    # P.show_analyzed_arrayrun(csv_filename=None, analysis='MeanFR', variable_unit='Hz', NG_id_list=['NG1', 'NG2']) # Empty NG_id_list for all groups
   
    # P.analyze_arrayrun(metadata_filename=None, analysis='Coherence', t_idx_start=0, t_idx_end=-1)
    # P.show_analyzed_arrayrun(csv_filename=None, analysis='Coherence', NG_id_list=['NG3']) 
    
    # P.analyze_arrayrun(metadata_filename=None, analysis='GrCaus', t_idx_start=0, t_idx_end=-1, **extra_GrCaus_attributes)
    # P.show_analyzed_arrayrun(csv_filename=None, analysis='GrCaus', NG_id_list=['NG3']) 

    # Classification is currently based on Granger causality F-statistics, thus gc is done first, and we need the extra attributes
    # Currently this gives only accuracy, i.e. N correct classifications / N all classifications
    # P.analyze_arrayrun(metadata_filename=None, analysis='Classify', t_idx_start=0, t_idx_end=-1, **extra_GrCaus_attributes)
    # P.show_analyzed_arrayrun(csv_filename=None, analysis='Classify') 
    
    ## System Profile ##
    '''
    The system_polar_bar method operates on output folder and its subfolders. It searches for csv files; only valid array 
    analysis csv files are allowed, others are cought to ValueError. 
    '''
    P.system_polar_bar(row_selection = [3, 15, 35], folder_name=None)

    plt.show()
