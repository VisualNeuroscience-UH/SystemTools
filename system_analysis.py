# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests as gc_test
from statsmodels.tsa.stattools import adfuller as stationarity_test

# Computational neuroscience
import brian2.units as b2u
import elephant as el 
from elephant.causality.granger import pairwise_granger, conditional_granger
from neo.core import AnalogSignal
import quantities as pq

# Builtin
import os
import pickle

# Current repo
from system_utilities import SystemUtilities


# Develop
import pdb
import warnings
warnings.filterwarnings("ignore")

'''
Module on analysis of simulated electrophysiology data.

Inherits SystemViz which inherits SystemUtilities

Developed by Simo Vanni 2020-2021
'''

class SystemAnalysis(SystemUtilities):

    map_analysis_names = {'meanfr':'MeanFR', 'eicurrentdiff':'EICurrentDiff', 'grcaus':'GrCaus'}
    map_data_types = {'meanfr':'spikes_all', 'eicurrentdiff':'vm_all', 'grcaus': 'vm_all'}

    def __init__(self, path='./'):

        self.path=path

    def _get_spikes_by_interval(self, data_by_group, t_idx_start, t_idx_end):
        spikes = data_by_group['t'][np.logical_and(data_by_group['t'] > t_idx_start * b2u.second, data_by_group['t'] < t_idx_end * b2u.second)]
        return spikes

    def  _analyze_meanfr(self, data, NG, t_idx_start, t_idx_end):

        data_by_group = data['spikes_all'][NG]
        # Get and mark MeanFR to df
        N_neurons = data_by_group['count'].size

        spikes = self._get_spikes_by_interval(data_by_group, t_idx_start=t_idx_start, t_idx_end=t_idx_end)

        dt = self._get_dt(data)

        MeanFR = spikes.size / (N_neurons * (t_idx_end - t_idx_start) * dt)

        return MeanFR

    def _get_currents_by_interval(self, data, NG, t_idx_start=0, t_idx_end=None):

        ge = data['ge_soma_all'][NG]['ge_soma']
        gi = data['gi_soma_all'][NG]['gi_soma']
        vm = data['vm_all'][NG]['vm']

        # Get necessary variables
        # Calculate excitatory, inhibitory (and leak) currents
        gl = data['Neuron_Groups_Parameters'][NG]['namespace']['gL']
        El = data['Neuron_Groups_Parameters'][NG]['namespace']['EL']         
        I_leak = gl * (El - vm)

        # If no driving force in neuron vm model synapses. This is currently denoted in neuron group name by _CI_ prefix
        if '_CI_' in NG : 
            I_e =  ge * b2u.mV
            I_i =  gi * b2u.mV
        else:
            Ee = data['Neuron_Groups_Parameters'][NG]['namespace']['Ee']
            Ei = data['Neuron_Groups_Parameters'][NG]['namespace']['Ei']         
            I_e =  ge * (Ee - vm)
            I_i =  gi * (Ei - vm)

        return I_e[t_idx_start:t_idx_end,:], I_i[t_idx_start:t_idx_end,:], I_leak[t_idx_start:t_idx_end,:]

    def  _analyze_eicurrentdiff(self, data, NG, t_idx_start=0, t_idx_end=None):

        I_e, I_i, I_leak = self._get_currents_by_interval(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)

        N_neurons = I_e.shape[1]

        # Calculate difference unit by unit, I_e all positive, I_i all negative, thus the difference is +
        EIdifference = I_e[t_idx_start:t_idx_end] + I_i[t_idx_start:t_idx_end]

        MeanEIdifference = np.sum(np.abs(EIdifference)) / N_neurons
 
        return MeanEIdifference
                
    def _get_vm_by_interval(self, data, NG, t_idx_start=0, t_idx_end=None):
        
        vm = data['vm_all'][NG]['vm']
        return vm[t_idx_start:t_idx_end,:]

    def _analyze_grcaus(self, data, source_signal, source_signal_dt, NG, t_idx_start=0, t_idx_end=None, test_stationarity=False):
        '''
        Get input and output timeseries.
        Run grangercausality for relevant pairs. 

        Stationary time series is a requirement for gc test. This is tested separately for each time
        series.

        The dt in is retained in case we need to have source and target at different sampling rates later.
        '''

        def _check_stationarity(signal, stationarity_p, time_lag, time_jump):
            stationary = False
            idx = 0
            while  not stationary:
                print(f'time lag: {time_lag} for idx {idx}\r', end="")
                if idx + 1 == signal.shape[1]:
                    stationary = True
                # Stationarity test for source signals
                st_result = stationarity_test(signal[:,idx], time_lag)
                p_value = st_result[1] 
                if np.isnan(p_value):
                    idx += 1
                    continue
                if p_value > stationarity_p:
                    time_lag += time_jump
                elif p_value <= stationarity_p:
                    idx += 1

            return time_lag

        # dt = self._get_dt(data)

        vm_unit = self._get_vm_by_interval(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)

        # Remove brian2 dimension for gc analysis
        target_signal = vm_unit / vm_unit.get_best_unit() 

        gc_matrix_np_F = np.full_like(np.empty((source_signal.shape[1], target_signal.shape[1])),np.nan)
        gc_matrix_np_p = np.full_like(np.empty((source_signal.shape[1], target_signal.shape[1])),np.nan)


        time_lag = 10
        time_jump = 20

        if test_stationarity is True:
            # Search for minimum valid time lag for this dataset
            stationarity_p = 0.05 # significance to reject non stationarity test
            
            time_lag = _check_stationarity(source_signal, stationarity_p, time_lag, time_jump)

            print(f'Source min time_lag {time_lag} time points')

            time_lag = _check_stationarity(target_signal, stationarity_p, time_lag, time_jump)

            print(f'Source and Target min time_lag {time_lag} time points')

        pre_idx_array =  np.arange(source_signal.shape[1])
        post_idx_array =  np.arange(target_signal.shape[1])

        # for each source signal, calculate all target signals.
        for pre_idx in pre_idx_array:
            for post_idx in post_idx_array:
                print(f'pre {pre_idx} post {post_idx}\r', end="")

                # The model order is the maximum number of lagged observations included in the model
                _source, _target = source_signal[:,pre_idx], target_signal[:,post_idx]
                signals = np.vstack([_target, _source]).T
                pairwise_gc_dict = gc_test(signals, time_lag, verbose=False)
                # TODO Get best F / p-value

                # dict_keys(['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'])
                # test statistic, pvalues, degrees of freedom
                gc_matrix_np_F[pre_idx, post_idx] = pairwise_gc_dict[time_lag][0]['ssr_ftest'][0]
                gc_matrix_np_p[pre_idx, post_idx] = pairwise_gc_dict[time_lag][0]['ssr_ftest'][1]

        # Find best matching target signal for each input signal
        gc_highest_F_idx = np.argmax(gc_matrix_np_F, axis=0)
        gc_highest_F = gc_matrix_np_F[gc_highest_F_idx] 
        gc_lowest_p = gc_matrix_np_p[gc_highest_F_idx] 
        # Calculate mean of best matching gc F values, one for each input signal
        MeanBestGrCaus = np.mean(gc_highest_F)
        # TODO Calculate CV of gc "grandmother index"

        # TODO Where is the best F / p value? Which time_lag to choose? First stationary?

        # Return one value per analysis (mean of best matching units), indicating GrCaus relation
        return MeanBestGrCaus, gc_highest_F_idx, gc_matrix_np_F, gc_matrix_np_p

    def get_analyzed_array_as_df(self, data_df, analysisHR=None, t_idx_start=0, t_idx_end=None):
    
        # Get neuron group names
        filename_0 = data_df['Full path'].values[0]
        data = self.getData(filename_0)
        NG_list = [n for n in data[self.map_data_types[analysisHR.lower()]].keys() if 'NG' in n]

        # Add neuron group columns
        for NG in NG_list:
            data_df[f'{analysisHR}_' + NG] = np.nan
        dt = self._get_dt(data)
        
        # Get duration
        if t_idx_end is None:
            t_idx_end = int(data['runtime']  / dt)

        # Get reference data for granger causality
        if analysisHR.lower() in ['grcaus']:
            analog_input = self.getData( self.input_filename, data_type=None)
            source_signal = analog_input['stimulus'].T # We want time x units
            source_signal_dt = analog_input['frameduration']
            target_group = self.NG_name


        # Loop through datafiles
        for this_index, this_file in zip(data_df.index, data_df['Full path'].values):
            data = self.getData(this_file)
            # Loop through neuron groups 
            if analysisHR.lower() in ['meanfr', 'eicurrentdiff']:
                for NG in NG_list:
                    # _analyze_meanfr or _analyze_eicurrentdiff, analysis by single group
                    analyzed_results = eval(f'self._analyze_{analysisHR.lower()}(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)')
                    data_df.loc[this_index,f'{analysisHR}_' + NG] = analyzed_results
            # _analyze__grcaus, analysis between two groups
            elif analysisHR.lower() in ['grcaus']:
                # check how multivariate gc is analyzed; are min, max, mean, median useful?
                # Apply this to _analyze_grangercausality
                MeanBestGrCaus, gc_highest_F_idx, gc_matrix_np_F, gc_matrix_np_p = eval(f'self._analyze_{analysisHR.lower()}(data, source_signal, source_signal_dt, target_group, t_idx_start=t_idx_start, t_idx_end=t_idx_end)')
                data_df.loc[this_index,f'{analysisHR}_' + NG] = MeanBestGrCaus

        return data_df

    def analyze_arrayrun(self, metadata_filename=None, analysis=None, t_idx_start=0, t_idx_end=None):
        '''
        Create mean firing rate csv table for array run. Needs a metadata file.
        '''
        # Map to standard camelcase
        assert analysis.lower() in self.map_analysis_names.keys(), 'Analysis type not found, aborting...'
        analysisHR = self.map_analysis_names[analysis.lower()]

        data_df = self.getData(metadata_filename, data_type='metadata')
        # data_df = self.getMeanFR_array(data_df, t_idx_start=t_idx_start, t_idx_end=t_idx_end)
        data_df = self.get_analyzed_array_as_df(data_df, analysisHR=analysisHR, t_idx_start=t_idx_start, t_idx_end=t_idx_end)

        # Drop Full path column for concise printing
        mean_df = data_df.drop(['Full path'], axis=1)

        # # Display values
        # self.pp_df_full(mean_df)

        # Replace metadata with scalar value for MeanFR or EICurrentDiff
        metadata_fullpath_filename = self._parsePath(metadata_filename, data_type='metadata')
        metadataroot, metadataextension = os.path.splitext(metadata_fullpath_filename)
        filename_out = metadataroot.replace('metadata', analysisHR)
        csv_name_out = filename_out + '.csv'
        mean_df.to_csv(csv_name_out, index=True)

if __name__=='__main__':

    path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'

    analysis = SystemAnalysis(path=path)
    NG_name = 'NG3_L4_SS2_L4'

    analysis.plot_readout_on_input(NG_name, normalize=False, filename='Replica_test_results_20210114_1750000.gz')    
    analysis.show_spikes(filename='Replica_test_results_20210114_1750000.gz')
    
    plt.show()