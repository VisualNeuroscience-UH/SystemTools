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

    def __init__(self, path='./'):

        self.path=path

    def _parsePath(self, filename, data_type='results'):
        '''
        This internal function returns full path to either 1) the file given by the filename at self.path 
        folder, or 2) to file with most recent modification time at self.path whose filename contains string 
        data_type.
        '''
        data_fullpath_filename = None
        path = self.path
        experiment_folder = self.experiment_folder

        if filename is None:
            data_fullpath_filename = self._most_recent(data_type=data_type)
        
        # Filename exists
        if not data_fullpath_filename:
            data_fullpath_filename = os.path.join(path,filename)

        # If not found in project path, try project/experiment folder -path
        if not os.path.isfile(data_fullpath_filename) :
            data_fullpath_filename = os.path.join(path, experiment_folder, filename)
            assert os.path.isfile(data_fullpath_filename), 'Could not parse filepath, aborting...'

        print(f'Working on file {data_fullpath_filename}')

        return data_fullpath_filename

    def _listdir_loop(self, path, data_type):
        files = []
        for f in os.listdir(path):
            if data_type in f:
                files.append(f)

        paths = [os.path.join(path, basename) for basename in files]

        return paths

    def _most_recent(self, data_type='results'):

        path = self.path

        paths = self._listdir_loop(path, data_type)

        if not paths and self.experiment_folder is not None:
            path_folder = os.path.join(path,self.experiment_folder)
            print(f'Did not find files of type {data_type} at {path}, trying at {path_folder}')
            paths = self._listdir_loop(path_folder, data_type)

        assert paths, f'No files of type {data_type} found at {path}, aborting...'
        data_fullpath_filename = max(paths, key=os.path.getctime)

        return data_fullpath_filename

    def _figsave(self, figurename='', myformat='png', suffix=''):
        
        # Saves current figure to working dir

        if myformat[0] == '.':
            myformat=myformat[1:]

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

    def getData(self, filename=None, data_type='results'):

        # Explore which is the most recent file in path of data_type and add full path to filename 
        data_fullpath_filename = self._parsePath(filename, data_type=data_type)

        # If extension is .gz, open pickle, else assume .mat
        filename_root, filename_extension = os.path.splitext(data_fullpath_filename)
        if 'gz' in filename_extension:
            fi = open(data_fullpath_filename, 'rb')
            try:
                data_pickle = zlib.decompress(fi.read())
                data = pickle.loads(data_pickle)
            except:
                with open(data_fullpath_filename, 'rb') as data_pickle:
                    data = pickle.load(data_pickle)
        elif 'mat' in filename_extension:
            data = {}
            sio.loadmat(data_fullpath_filename,data) 
        else:
            raise TypeError('U r trying to input unknown filetype, aborting...')

        # print(f'Acquiring data from {data_fullpath_filename}')
        return data

    def close(self):
        plt.close('all')

    def showVm(self, filename=None, savefigname=''):
        # Shows data on filename. If filename remains None, shows the most recent data.

        data = self.getData(filename, data_type='results')

        # Visualize
        # Extract connections from data dict
        list_of_results = [n for n in data['vm_all'].keys() if 'NG' in n]

        print(list_of_results)
        n_images=len(list_of_results)
        n_columns = 2
        n_rows = int(np.ceil(n_images/n_columns))

        t=data['vm_all'][list_of_results[0]]['t']

        fig, axs = plt.subplots(n_rows, n_columns)
        axs = axs.flat

        for ax, results in zip(axs,list_of_results):
            N_monitored_neurons = data['vm_all'][results]['vm'].shape[1]
            N_neurons = len(data['positions_all']['w_coord'][results])

            im = ax.plot(t, data['vm_all'][results]['vm'])
            ax.set_title(results, fontsize=10)

        if savefigname:
            self._figsave(figurename=savefigname)

    def pp_df_full(self, df):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            print(df)

    def pp_df_memory(self, df):
        BYTES_TO_MB_DIV = 0.000001
        mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
        print("Memory usage is " + str(mem) + " MB")

    def pp_obj(self, obj):
        from IPython.lib.pretty import pprint 
        pprint(obj)
        print(f'\nObject size is {sys.getsizeof(obj)} bytes')

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

    def showConnections(self, filename=None, hist_from=None, savefigname=''):

        data = self.getData(filename, data_type='connections')

        # Visualize
        # Extract connections from data dict
        list_of_connections = [n for n in data.keys() if '__to__' in n]
    
        # Pick histogram data
        if hist_from is None:
            hist_from = list_of_connections[-1]

        print(list_of_connections)
        n_images=len(list_of_connections)
        n_columns = 2
        n_rows = int(np.ceil(n_images/n_columns))
        fig, axs = plt.subplots(n_rows, n_columns)
        axs = axs.flat
        for ax, connection in zip(axs,list_of_connections):
            im = ax.imshow(data[connection]['data'].todense())
            ax.set_title(connection, fontsize=10)
            fig.colorbar(im, ax=ax)
        data4hist = np.squeeze(np.asarray(data[hist_from]['data'].todense().flatten()))
        data4hist_nozeros = np.ma.masked_equal(data4hist,0)
        Nzeros = data4hist==0
        proportion_zeros = Nzeros.sum() / Nzeros.size
        axs[(n_rows * n_columns)-1].hist(data4hist_nozeros)
        axs[(n_rows * n_columns)-1].set_title(f"{hist_from}\n{(proportion_zeros * 100):.1f}% zeros (not shown)")
        if savefigname:
            self._figsave(figurename=savefigname)

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

    def showI(self, filename=None, savefigname='', neuron_index=None):

        data = self.getData(filename, data_type='results')
        # Visualize
        # Extract connections from data dict
        list_of_results_ge = [n for n in data['ge_soma_all'].keys() if 'NG' in n]
        list_of_results_gi = [n for n in data['gi_soma_all'].keys() if 'NG' in n]
        list_of_results_vm = [n for n in data['vm_all'].keys() if 'NG' in n]

        assert list_of_results_ge == list_of_results_gi == list_of_results_vm, 'Some key results missing, aborting...'
        NG_list = list_of_results_ge 

        n_images = len(NG_list)
        n_columns = 2
        n_rows = int(np.ceil(n_images/n_columns))

        t = data['ge_soma_all'][NG_list[0]]['t']
        # time_idx_interval=[2000, 4000]
        time_idx_interval = [0, len(t)-1]

        fig, axs = plt.subplots(n_rows, n_columns)
        axs = axs.flat

        
        for ax, NG in zip(axs, NG_list):

            N_monitored_neurons = data['vm_all'][NG]['vm'].shape[1]
            N_neurons = len(data['positions_all']['w_coord'][NG])

            ge = data['ge_soma_all'][NG]['ge_soma']
            gi = data['gi_soma_all'][NG]['gi_soma']
            vm = data['vm_all'][NG]['vm']

            # pdb.set_trace()

            # Get necessary variables
            # Calculate excitatory, inhibitory (and leak) currents
            gl = data['Neuron_Groups_Parameters'][NG]['namespace']['gL']
            El = data['Neuron_Groups_Parameters'][NG]['namespace']['EL']         
            I_leak = gl * (El - vm)
            if (NG == 'NG1_L4_CI_SS_L4') or (NG == 'NG2_L4_CI_BC_L4') :
                I_e =  ge * b2u.mV
                I_i =  gi * b2u.mV
            elif NG == 'NG3_L4_SS2_L4' :
                Ee = data['Neuron_Groups_Parameters'][NG]['namespace']['Ee']
                Ei = data['Neuron_Groups_Parameters'][NG]['namespace']['Ei']         
                I_e =  ge * (Ee - vm)
                I_i =  gi * (Ei - vm)

            if neuron_index is None:
                neuron_index = {'NG1_L4_CI_SS_L4' : 0, 'NG2_L4_CI_BC_L4' : 0, 'NG3_L4_SS2_L4' : 0}

            # pdb.set_trace()
            ax.plot(    t[time_idx_interval[0]:time_idx_interval[1]], 
                        np.array([I_e[time_idx_interval[0]:time_idx_interval[1],neuron_index[NG]],
                        I_i[time_idx_interval[0]:time_idx_interval[1],neuron_index[NG]]]).T
                        )
            ax.legend(['I_e', 'I_i'])
            ax.set_title(NG + ' current', fontsize=10)

            # I_total_mean = np.mean(I_total[time_idx_interval[0]:time_idx_interval[1]] / b2u.namp)
            # I_total_mean_str = f'mean I = {I_total_mean:6.2f} nAmp'
            # ax.text(0.05, 0.95, I_total_mean_str, fontsize=10, verticalalignment='top', transform=ax.transAxes)

        if savefigname:
            self._figsave(figurename=savefigname)

    def _getI(self, neuron_group,data):

        # Extract connections from data dict
        # list_of_results_ge = [n for n in data['ge_soma_all'].keys() if 'NG' in n]
        # list_of_results_gi = [n for n in data['gi_soma_all'].keys() if 'NG' in n]
        # list_of_results_vm = [n for n in data['vm_all'].keys() if 'NG' in n]
        list_of_results_ge = [n for n in data['ge_soma_all'].keys() if neuron_group in n]
        list_of_results_gi = [n for n in data['gi_soma_all'].keys() if neuron_group in n]
        list_of_results_vm = [n for n in data['vm_all'].keys() if neuron_group in n]
        assert list_of_results_ge == list_of_results_gi == list_of_results_vm, "different N neuron groups monitored, can not calculate current, aborting..."

        if len(list_of_results_ge)==0:
            #No currents monitored for thir group
            I_total_mean = 0 # to avoid stupid error later
            I_excitatory_mean = 0
            I_inhibitory_mean = 0
            return I_total_mean, I_excitatory_mean, I_inhibitory_mean
        # These data are available in physiology_configuration file (inside results datafile), but hard coded here for the time being
        El = -65 * b2u.mV
        gl = 50 * b2u.nS
        Ee = 0 * b2u.mV
        Ei = -75 * b2u.mV

        time_idx_interval=[2000, 4000]

        for results_ge, results_gi, results_vm in zip(  list_of_results_ge, \
                                                            list_of_results_gi,list_of_results_vm):

            N_monitored_neurons = data['vm_all'][results_vm]['vm'].shape[1]
            N_neurons = len(data['positions_all']['w_coord'][results_vm])

            ge= data['ge_soma_all'][results_ge]['ge_soma']
            gi= data['gi_soma_all'][results_gi]['gi_soma']
            vm= data['vm_all'][results_vm]['vm']
            I_total = gl * (El - vm) + ge * (Ee - vm) + gi * (Ei - vm)
            I_excitatory = ge * (Ee - vm)
            I_inhibitory = gi * (Ei - vm)

            if N_monitored_neurons == N_neurons: 
                neuron_index = self._getNeuronIndex(data, results_vm, position=0+0j)
            I_total_mean = np.mean(I_total[time_idx_interval[0]:time_idx_interval[1]] / namp)
            I_excitatory_mean = np.mean(I_excitatory[time_idx_interval[0]:time_idx_interval[1]] / namp)
            I_inhibitory_mean = np.mean(I_inhibitory[time_idx_interval[0]:time_idx_interval[1]] / namp)
            
        return I_total_mean, I_excitatory_mean, I_inhibitory_mean

    def showSpectra(self, filename=None, savefigname=''):

        data = self.getData(filename, data_type='results')

        # Visualize
        coords='w_coord'
        # Extract connections from data dict
        list_of_results = [n for n in data['spikes_all'].keys() if 'NG' in n]

        print(list_of_results)
        n_images=len(list_of_results)
        n_columns = 2
        nbins=400
        duration = data['runtime']
        bar_width = duration / nbins
        freqCutoff = 200 # Hz
        ylims=np.array([-18.4,18.4])
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