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
Module on analysis of simulated electrophysiology data

Developed by Simo Vanni 2020-2021
'''

class SystemAnalysis(SystemUtilities):

    def __init__(self, path='./'):

        self.path=path

    def _get_spikes_by_interval(self, data_by_group, time_start=0, time_end=None):
        spikes = data_by_group['t'][
            np.logical_and(data_by_group['t'] > time_start * b2u.second, data_by_group['t'] < time_end * b2u.second)]
        return spikes

    def showSpikes(self, filename=None, savefigname=''):

        data = self.getData(filename, type='results')

        # Visualize
        # Extract connections from data dict
        list_of_results = [n for n in data['spikes_all'].keys() if 'NG' in n]

        print(list_of_results)
        n_images=len(list_of_results)
        n_columns = 2
        n_rows = int(np.ceil(n_images/n_columns))

        fig, axs = plt.subplots(n_rows, n_columns)
        axs = axs.flat

        for ax, results in zip(axs,list_of_results):

            im = ax.plot(data['spikes_all'][results]['t'], data['spikes_all'][results]['i'],'.')
            ax.set_title(results, fontsize=10)

        if savefigname:
            self._figsave(figurename=savefigname)

        plt.show()


    def getMeanFR(self, data_df, time_start=0, time_end=None):
    
        # Get neuron group names
        filename_0 = data_df['Full path'].values[0]
        data = self.getData(filename_0)
        list_of_group_names = [n for n in data['spikes_all'].keys() if 'NG' in n]

        # Add neuron group columns
        for this_group in list_of_group_names:
            data_df['MeanFR_' + this_group] = np.nan

        # Get duration
        duration = data['runtime'] # unitless scalar, in seconds
        assert time_start < duration, f'Duration is {duration} (seconds) but time start is {time_start}'
        # Determine start and end times of spike rate averaging
        if time_end is None:
            time_end = duration

        # Loop through datafiles
        for this_index, this_file in zip(data_df.index, data_df['Full path'].values):
            data = self.getData(this_file)
            # Loop through neuron groups 
            for this_group in list_of_group_names:
                data_by_group = data['spikes_all'][this_group]
                # Get and mark MeanFR to df
                N_neurons = data_by_group['count'].size

                spikes = self._get_spikes_by_interval(data_by_group, time_start=time_start, time_end=time_end)

                MeanFR = spikes.size / (N_neurons * (time_end - time_start))
                data_df.loc[this_index,'MeanFR_' + this_group] = MeanFR

        return data_df

    def printMeanFR(self, filename=None, time_start=0, time_end=None):

        data_df = self.getData(filename, type='metadata')
    
        data_df = self.getMeanFR(data_df, time_start=time_start, time_end=time_end)

        # Drop Full path column for concise printing
        mean_df = data_df.drop(['Full path'], axis=1)

        # Display values
        self.pp_df_full(mean_df)

        # Replace meatadata with MeanFR
        metadata_fullpath_filename = self._parsePath(filename, type='metadata')
        metadataroot, metadataextension = os.path.splitext(metadata_fullpath_filename)
        filename_out = metadataroot.replace('metadata', 'MeanFR')
        csv_name_out = filename_out + '.csv'
        mean_df.to_csv(csv_name_out, index=True)

if __name__=='__main__':

    path = r'C:\Users\Simo\Laskenta\SimuOut\Tomas_021220\oP'

    analysis = SystemAnalysis(path=path)

    analysis.printMeanFR(filename=None, time_start=0, time_end=None)
