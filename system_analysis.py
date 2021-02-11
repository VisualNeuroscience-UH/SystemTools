# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Computational neuroscience
import brian2.units as b2u
import elephant as el

# Builtin
import os
import pickle

# Current repo
from system_utilities import SystemUtilities


# Develop
import pdb

'''
Module on analysis of simulated electrophysiology data.

Inherits SystemViz which inherits SystemUtilities

Developed by Simo Vanni 2020-2021
'''

class SystemAnalysis(SystemUtilities):

    map_analysis_names = {'meanfr':'MeanFR', 'eicurrentdiff':'EICurrentDiff'}
    map_data_types = {'meanfr':'spikes_all', 'eicurrentdiff':'vm_all'}

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


        # Loop through datafiles
        for this_index, this_file in zip(data_df.index, data_df['Full path'].values):
            data = self.getData(this_file)
            # Loop through neuron groups 
            for NG in NG_list:

                # _analyze_meanfr or _analyze_eicurrentdiff
                analyzed_results = eval(f'self._analyze_{analysisHR.lower()}(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)')

                data_df.loc[this_index,f'{analysisHR}_' + NG] = analyzed_results

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

        # Replace metadata with MeanFR
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