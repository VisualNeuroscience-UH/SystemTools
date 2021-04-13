# Analysis
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import pandas as pd

# Computational neuroscience
# from cxsystem2.core.tools import write_to_file as wtf
import brian2.units as b2u
import elephant as el
from neo.core import AnalogSignal
import quantities as pq

# Builtin
import zlib
import pickle
import os
import sys

# Develop
import pdb

'''
Module on basic utilities. Base class for other system modules in the same repo.

Developed by Simo Vanni 2020-2021
'''

class SystemUtilities():
    
    # Types of data in simulation outputfolder
    data_types_out = ['results', 'connections', 'metadata', 'meanfr', 'eicurrentdiff', 'grcaus', 'meanvm']

    def __init__(self, path='./'):

        self.path=path

        # Init some variables
        self.input_folder = None
        self.output_folder = None

    def round_to_n_significant(self, value_in, significant_digits=2):

        if value_in != 0 and not np.isnan(value_in):
            int_to_subtract = significant_digits - 1
            value_out = round(value_in, -int(np.floor(np.log10(np.abs(value_in))) - int_to_subtract))
        else:
            value_out = value_in
            
        return value_out    

    def _check_cadidate_file(self, path, filename):
        candidate_data_fullpath_filename = os.path.join(path, filename)
        if os.path.isfile(candidate_data_fullpath_filename):
            data_fullpath_filename = candidate_data_fullpath_filename
            return data_fullpath_filename
        else:
            return None

    def _parsePath(self, filename, data_type=None):
        '''
        This internal function returns full path to either given filename or to most recently
        updated file of given data_type (a.k.a. containing key substring in filename). 
        Note that the data_type can be timestamp.
        '''
        data_fullpath_filename = None
        path = self.path
        input_folder = self.input_folder
        output_folder = self.output_folder
        if output_folder is not None:
            output_path = os.path.join(path, output_folder)
        else:
            # Set current path if run separately from project 
            output_path = os.path.join(path, './')
        if input_folder is not None:
            input_path = os.path.join(path, input_folder)
        else:
            input_path = os.path.join(path, './')

        # Check first for direct load in current directory. E.g. for direct ipython testing
        if filename:
            data_fullpath_filename = self._check_cadidate_file('./', filename)

        # Next check direct load in output path, input path and project path in this order
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_cadidate_file(output_path, filename)
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_cadidate_file(input_path, filename)
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_cadidate_file(path, filename)
    
        # Parse output folder for given data_type
        elif data_type in self.data_types_out:
            data_fullpath_filename = self._most_recent(output_path, data_type=data_type)
            if not data_fullpath_filename:
                raise FileNotFoundError(f'Did not find {data_type} file in folder {output_path}')
            
        # Parse data_type next in project/input and project paths
        elif data_type is not None:
            # Check for data_type first in input folder
            data_fullpath_filename = self._most_recent(input_path, data_type=data_type)
            # Check for data_type next in project folder
            if not data_fullpath_filename:
                data_fullpath_filename = self._most_recent(path, data_type=data_type)

        assert data_fullpath_filename is not None, 'Could not parse filepath, aborting...'

        print(f'Working on file {data_fullpath_filename}')

        # For easy access
        self.most_recent_loaded_file = data_fullpath_filename 

        return data_fullpath_filename

    def _listdir_loop(self, path, data_type):
        files = []
        for f in os.listdir(path):
            if data_type in f.lower():
                files.append(f)

        paths = [os.path.join(path, basename) for basename in files]

        return paths

    def _most_recent(self, path, data_type=None):

        paths = self._listdir_loop(path, data_type)

        if not paths:
            return None
        else:
            data_fullpath_filename = max(paths, key=os.path.getctime)
            return data_fullpath_filename

    def _figsave(self, figurename='', myformat='png', suffix=''):
        
        # Saves current figure to working dir

        if myformat[0] == '.':
            myformat = myformat[1:]

        filename, file_extension = os.path.splitext(figurename)

        filename = filename + suffix

        if not file_extension:
            file_extension = '.' + myformat

        if not figurename:
            figurename = 'MyFigure' + file_extension
        else:
            figurename = filename + file_extension

        path = self.path
        figurename_fullpath = os.path.join(path, figurename)
        plt.savefig(figurename_fullpath, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=file_extension[1:],
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)

    def getData(self, filename=None, data_type=None):
        '''
        Open requested file and get data.
        '''
        if data_type is not None:
            data_type = data_type.lower()
        # Explore which is the most recent file in path of data_type and add full path to filename 
        data_fullpath_filename = self._parsePath(filename, data_type=data_type)
        # If extension is .gz, open pickle, else assume .mat
        filename_root, filename_extension = os.path.splitext(data_fullpath_filename)
        if 'gz' in filename_extension:           
            try:
                fi = open(data_fullpath_filename, 'rb')
                data_pickle = zlib.decompress(fi.read())
                data = pickle.loads(data_pickle)
            except:
                with open(data_fullpath_filename, 'rb') as data_pickle:
                    data = pickle.load(data_pickle)
        elif 'mat' in filename_extension:
            data = {}
            sio.loadmat(data_fullpath_filename,data) 
        elif 'csv' in filename_extension:
            data = pd.read_csv(data_fullpath_filename)
        else:
            raise TypeError('U r trying to input unknown filetype, aborting...')

        # print(f'Acquiring data from {data_fullpath_filename}')
        return data

    def _get_dt(self, data):
        dt = (data['time_vector'][1] - data['time_vector'][0]) / b2u.second
        return dt

    def _get_nsamples(self, data):
        nsamples = len(data['time_vector'])
        return nsamples

    def close(self):
        plt.close('all')

    def pp_df_full(self, df):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1): 
            print(df)

    def pp_df_memory(self, df):
        BYTES_TO_MB_DIV = 0.000001
        mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
        print("Memory usage is " + str(mem) + " MB")

    def pp_obj(self, obj):
        from IPython.lib.pretty import pprint 
        pprint(obj)
        print(f'\nObject size is {sys.getsizeof(obj)} bytes')

    def get_added_attributes(self, obj1, obj2):

        XOR_attributes = set(dir(obj1)).symmetric_difference(dir(obj2))
        unique_attributes_list = [n for n in XOR_attributes if not n.startswith('_')]
        return unique_attributes_list

    def pp_attribute_types(self, obj, attribute_list=[]):

        if not attribute_list:
            attribute_list = dir(obj)

        for this_attribute in attribute_list:
            attribute_type = eval(f'type(obj.{this_attribute})')
            print(f'{this_attribute}: {attribute_type}')

    def read_neo_block(self, filename):
        import neo
        reader = neo.io.NixIO(filename=filename, mode='ro')
        block = reader.read_block() 
        return block

    def _getDistance(self, positions,index_position=None):
        '''Calculates distances between neurons. If index position is given, 
        only distances to this neuron is calculated. Without index position, 
        all distance pairs will be calculated.
        Assumes positions as a list or numpy array of complex coordinates'''
        
        positions_array = np.asarray(positions)
        assert len(positions_array) > 1, 'At least two positions necessary for distance'

        # Calculate distance btw one cell and all other cells
        def dist(index_position,other_positions):
            # Assumes arrays of complex numbers
            d = np.sqrt((np.real(index_position)-np.real(other_positions))**2 + 
                (np.imag(index_position)-np.imag(other_positions))**2)
            return d

        # Check whether index_position exists
        if index_position is not None:
            # Calculate distance between index neuron and other neurons
            distance = dist(index_position,positions_array)
        # If only two positions are given
        elif len(positions_array)==2:
            distance = dist(positions_array[0],positions_array[1])
        # Otherwise
        else:
            # Init result matrix
            distance = np.zeros([len(positions_array),len(positions_array)])

            # Loop index neuron positions
            for idx, index_position in enumerate(positions_array):
                ## Create vector array of other positions
                #other_positions = np.hstack([positions_array[:idx],positions_array[idx+1:]]) 
                # Calculate distance between index neuron and all neurons including itself
                distance[idx,:] = dist(index_position,positions_array)
            
        return distance

    def _getNeuronIndex(self, data, neuron_group, position=0+0j):
        neuron_index=data['positions_all']['w_coord'][neuron_group].index(position)
        return neuron_index

    def showG(self, filename=None, savefigname=''):

        data = self.getData(filename, data_type='results')
        # Visualize
        # Extract connections from data dict
        list_of_results_ge = [n for n in data['ge_soma_all'].keys() if 'NG' in n]
        list_of_results_gi = [n for n in data['gi_soma_all'].keys() if 'NG' in n]

        print(list_of_results_ge)
        n_images=len(list_of_results_ge)
        n_columns = 2
        n_rows = int(np.ceil(n_images/n_columns))

        t=data['ge_soma_all'][list_of_results_ge[0]]['t']
        time_idx_interval=[2000, 4000]

        fig, axs = plt.subplots(n_rows, n_columns)
        axs = axs.flat

        for ax, results in zip(axs,list_of_results_ge):
            im = ax.plot(t[time_idx_interval[0]:time_idx_interval[1]], 
                        data['ge_soma_all'][results]['ge_soma'][time_idx_interval[0]:time_idx_interval[1],0:-1:20])
            ax.set_title(results + ' ge', fontsize=10)

        fig2, axs2 = plt.subplots(n_rows, n_columns)
        axs2 = axs2.flat

        for ax2, results2 in zip(axs2,list_of_results_gi):
            im = ax2.plot(t[time_idx_interval[0]:time_idx_interval[1]], 
                        data['gi_soma_all'][results2]['gi_soma'][time_idx_interval[0]:time_idx_interval[1],0:-1:20])
            ax2.set_title(results2 + ' gi', fontsize=10)

        if savefigname:
            self._figsave(figurename=savefigname)

    def showSpectra(self, filename=None, savefigname=''):

        data = self.getData(filename, data_type='results')

        # Visualize
        coords='w_coord'
        # Extract connections from data dict
        list_of_results = [n for n in data['spikes_all'].keys() if 'NG' in n]

        print(list_of_results)
        n_images = len(list_of_results)
        n_columns = 2
        nbins = 400
        duration = data['runtime']
        bar_width = duration / nbins
        freqCutoff = 200 # Hz
        ylims = np.array([-18.4,18.4])
        n_rows = int(np.ceil(n_images/n_columns))

        width_ratios = np.array([2,2])

        fig, axs = plt.subplots(n_rows * 2, n_columns, gridspec_kw={'width_ratios': width_ratios})
        axs = axs.flat

        for ax1, results in zip(axs[0:-1:2],list_of_results):
            # im = ax1.scatter(data['spikes_all'][results]['t'], data['spikes_all'][results]['i'],s=1)
            counts, bins = np.histogram(data['spikes_all'][results]['t']/b2u.second, range=[0,duration], bins=nbins)
            Nneurons = data['number_of_neurons'][results]
            firing_rates = counts * (nbins / (duration * Nneurons))
            im = ax1.bar(bins[:-1], firing_rates, width=bar_width)
            ax1.set_title(results, fontsize=10)

        for ax2, results in zip(axs[1:-1:2],list_of_results):
            try:
                Vms=data['vm_all'][results]['vm']
            except:
                continue
            sigarr = AnalogSignal(Vms, units='mV', sampling_rate=10000*pq.Hz)
            freqs, psd = el.spectral.welch_psd(sigarr, n_segments=8, len_segment=None, frequency_resolution=None, fs=10000, nfft=None, scaling='density', axis=- 1)
            
            # Cut to desired freq range
            freqs_for_plotting = freqs[freqs < freqCutoff]
            psd_for_plotting = psd[:,freqs < freqCutoff]

            # Calculate mean psd
            psd_for_plotting_mean = np.mean(psd_for_plotting, axis=0)
    
            ax2.plot(freqs_for_plotting, psd_for_plotting.T, color='gray', linewidth=.25)
            ax2.plot(freqs_for_plotting, psd_for_plotting_mean, color='black', linewidth=1)

        # Shut down last pair if uneven n images
        if np.mod(n_images,2):
            axs[-2].axis('off')
            axs[-1].axis('off')
        if savefigname:
            self._figsave(figurename=savefigname)

if __name__=='__main__':
    pass
    # path = r'C:\Users\Simo\Laskenta\Models\Grossberg\CxPytestWorkspace\all_results'
    # os.chdir(path)

    # SU = SystemUtilities(path=path)

    # SU.showSpectra(filename='results_low_500NG4-3_results_20200826_1153084.gz', savefigname='results_low_500NG4-3.eps')