# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import colour 


# Analysis
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Builtin
import os
import copy

# Computational neuroscience
import brian2.units as b2u

# Current repo
from system_analysis import SystemAnalysis

# Develop
import pdb

'''
Module on visualization

Developed by Simo Vanni 2020-2021
'''

class SystemViz(SystemAnalysis):

    cmap='gist_earth'

    def unpivot_dataframe(self, wide_df, index_column=None, kw_sub_to_unpivot=None):

        # find columns to unpivot
        columns_to_unpivot = []
        short_var_names = []
        for column_name in wide_df.columns:
            if kw_sub_to_unpivot in column_name:
                columns_to_unpivot.append(column_name)
                short_var_names.append(column_name.replace(f'{kw_sub_to_unpivot}_',''))

        rename_dict = {a:b for a,b in zip(columns_to_unpivot, short_var_names)}  

        wide_df_rename = wide_df.rename(columns=rename_dict)

        long_df = pd.melt(  wide_df_rename, 
                            id_vars=index_column, 
                            value_vars=short_var_names, 
                            var_name='groupName', 
                            value_name=kw_sub_to_unpivot)

        return long_df
        
    def pivot_to_2d_dataframe(self, long_df, index_column_name= None, column_column_name=None, value_column_name=None):

        assert all([index_column_name, column_column_name, value_column_name]), 'Missing some column names, aborting...'

        wide_2d_df = long_df.pivot(index=index_column_name, columns=column_column_name, values=value_column_name)

        return wide_2d_df

    def _build_columns(self, data_dict, new_data, Ntimepoints, datatype):

        tmp_dict = copy.deepcopy(data_dict)
        # columns for input
        new_data_shape = np.asarray(new_data.shape)
        new_data_time_dim = np.where(new_data_shape == Ntimepoints)
        if new_data_time_dim[0] == 1:
            new_data = new_data.T
            new_data_shape = np.asarray(new_data.shape)

        for ch_idx in np.arange(new_data_shape[1]):
            ch_name = f'{datatype}_ch_{ch_idx}'
            ch_data = new_data[:,ch_idx]
            tmp_dict[ch_name] = ch_data

        dims = new_data_shape[1]
        return tmp_dict, dims

    def _string_on_plot(self, ax, variable_name=None, variable_value=None, variable_unit=None):

        plot_str = f'{variable_name} = {variable_value:6.2f} {variable_unit}'
        ax.text(0.05, 0.95, plot_str, fontsize=8, verticalalignment='top', transform=ax.transAxes,
        bbox=dict(boxstyle="Square,pad=0.2", fc="white", ec="white", lw=1))

    def show_readout_on_input(self, results_filename=None, normalize=False, unit_idx_list=None):
        '''
        Get input, get data. Scaling. turn to df, format df, Plot curves.
        '''
        # Get data and input
        data = self.getData(filename=results_filename, data_type='results')

        analog_signal =  self.read_input_matfile(filename=self.input_filename, variable='stimulus')

        NG_name = self.NG_name
        t = data['vm_all'][NG_name]['t'] # All timesteps
        assert t.ndim == 1, 'timepoints are not a 1-dim vector, aborting...'

        data_vm = data['vm_all'][NG_name]['vm']

        if unit_idx_list is not None:
            # Get a subsample of units
            idx = np.asarray(unit_idx_list)
            analog_signal = analog_signal[:,idx]
            data_vm = data_vm[:,idx]

        if normalize==True:
            analog_signal = (analog_signal - np.min(analog_signal)) / np.ptp(analog_signal)

        # Create dict and column for timepoints
        id_var = 't'
        data_dict = {id_var:t}
        Ntimepoints = t.shape[0]

        # columns for input
        data_dict_in, dims_IN = self._build_columns(data_dict, analog_signal, Ntimepoints, 'IN')

        # columns for vm
        data_dict_vm, dims_Dec_vm = self._build_columns(data_dict, data_vm, Ntimepoints, 'Dec_vm')

        # # Get final output dimension, to get values for unpivot
        # prod_dims = dims_IN * dims_Dec_vm

        df_from_arr_in = pd.DataFrame(data=data_dict_in)
        df_from_arr_vm = pd.DataFrame(data=data_dict_vm)

        value_vars_in = df_from_arr_in.columns[df_from_arr_in.columns != id_var]
        value_vars_vm = df_from_arr_vm.columns[df_from_arr_vm.columns != id_var]
        df_from_arr_unpivot_in = pd.melt(  df_from_arr_in, 
                                        id_vars=[id_var], 
                                        value_vars=value_vars_in, 
                                        var_name='units_in', 
                                        value_name='data_in')

        df_from_arr_unpivot_vm = pd.melt(  df_from_arr_vm, 
                                        id_vars=[id_var], 
                                        value_vars=value_vars_vm, 
                                        var_name='units_vm', 
                                        value_name='data_vm')

        sns.lineplot(x="t", y='data_in', data=df_from_arr_unpivot_in, hue='units_in', palette = 'dark')
        plt.legend(loc='upper left')
        ax2 = plt.twinx()
        sns.lineplot(x="t", y='data_vm', data=df_from_arr_unpivot_vm, palette = 'bright', hue='units_vm', ax=ax2)
        plt.legend(loc='upper right')

        if normalize==True:
            EL = data['Neuron_Groups_Parameters'][NG_name]['namespace']['EL'] 
            VT = data['Neuron_Groups_Parameters'][NG_name]['namespace']['VT']
            plt.ylim(EL, VT)

    def lineplot(self, data):
        data_df = pd.DataFrame(data)
        sns.lineplot(data=data_df)
        plt.show()

    def _show_coherence_of_two_signals(self, f, Cxy, Pwelch_spec_x, Pwelch_spec_y, Pxy, lags, \
        corr, latency, x, y, x_scaled, y_scaled):

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
        
        # ax1.semilogy(f,Cxy)
        ax1.plot(f,Cxy)
        ax1.set_xlabel('frequency [Hz]')
        ax1.set_ylabel('Coherence')
        ax1.set_title('Coherence')

        # ax2.semilogy(f,Pwelch_spec_x/np.max(Pwelch_spec_x))
        ax2.plot(f,Pwelch_spec_x/np.max(Pwelch_spec_x))
        # ax2.semilogy(f,Pwelch_spec_y/np.max(Pwelch_spec_y))
        ax2.plot(f,Pwelch_spec_y/np.max(Pwelch_spec_y))
        ax2.set_xlabel('frequency [Hz]')
        ax2.set_ylabel('PSD')
        ax2.set_title('Scaled power spectra')

        # ax3.semilogy(f,np.abs(Pxy))
        ax3.plot(f,np.abs(Pxy))
        ax3.set_xlabel('frequency [Hz]')
        ax3.set_ylabel('CSD [V**2/Hz]')
        ax3.set_title('Cross spectral density')

        ax4.plot(lags, corr)
        ax4.set_xlabel('samples')
        ax4.set_ylabel('correlation')
        ax4.set_title('Cross correlation')

        ax5.plot(x)
        ax5.plot(y)
        ax5.set_xlabel('time (samples)')
        ax5.set_ylabel('signal')
        ax5.set_title('Original signals')

        ax6.plot(x_scaled)
        ax6.plot(y_scaled)
        ax6.set_xlabel('time (samples)')
        ax6.set_ylabel('signal')
        ax6.set_title('Zero mean - unit variance signals')

        self._string_on_plot(ax4, variable_name='Latency', variable_value=latency, variable_unit='s')

    def show_spikes(self, results_filename=None, savefigname='', t_idx_start=0, t_idx_end=10000):

        data = self.getData(filename=results_filename, data_type='results')

        # Visualize
        # Extract connections from data dict
        NG_list = [n for n in data['spikes_all'].keys() if 'NG' in n]

        print(NG_list)
        fig, axs = self._prep_group_figure(NG_list)

        for ax, this_group in zip(axs,NG_list):

            im = ax.plot(data['spikes_all'][this_group]['t'], data['spikes_all'][this_group]['i'],'.')
            ax.set_title(this_group, fontsize=10)

            MeanFR = self._analyze_meanfr(data, this_group, t_idx_start=t_idx_start, t_idx_end=t_idx_end)
            self._string_on_plot(ax, variable_name='Mean FR', variable_value=MeanFR, variable_unit='Hz')

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_vm(self, results_filename=None, savefigname=''):
        # Shows data on filename. If filename remains None, shows the most recent data.

        data = self.getData(filename=results_filename, data_type='results')

        # Visualize
        # Extract connections from data dict
        NG_list = [n for n in data['vm_all'].keys() if 'NG' in n]

        print(NG_list)
 
        t=data['vm_all'][NG_list[0]]['t']

        fig, axs = self._prep_group_figure(NG_list)

        for ax, results in zip(axs,NG_list):
            N_monitored_neurons = data['vm_all'][results]['vm'].shape[1]
            # N_neurons = len(data['positions_all']['w_coord'][results])

            im = ax.plot(t, data['vm_all'][results]['vm'])
            ax.set_title(results, fontsize=10)

        if savefigname:
            self._figsave(figurename=savefigname)

    def _get_n_neurons_and_data_array(self, data, this_group, param_name, neuron_index=None):

        assert neuron_index is None or (isinstance(neuron_index, dict) and \
        (isinstance(neuron_index[this_group], int) or \
        isinstance(neuron_index[this_group], list))), \
        ''' neuron index for each group must be either None, dict[group_name]=int  
        eg {"NG1_L4_CI_SS_L4" : 150} or dict[group_name]=list of ints'''

        if neuron_index is None:
            N_monitored_neurons = data[f'{param_name}_all'][this_group][f'{param_name}'].shape[1]
            this_data =  data[f'{param_name}_all'][this_group][f'{param_name}']
        elif isinstance(neuron_index[this_group], int):
            N_monitored_neurons = 1
            this_data =  data[f'{param_name}_all'][this_group][f'{param_name}'][:,neuron_index[this_group]]
        elif isinstance(neuron_index[this_group], list):
            N_monitored_neurons = len(neuron_index[this_group])
            this_data =  data[f'{param_name}_all'][this_group][f'{param_name}'][:,neuron_index[this_group]]

        return N_monitored_neurons, this_data

    def show_analog_results(self, results_filename=None, savefigname='',param_name=None,startswith=None, neuron_index=None):
        # Shows data on filename. If filename remains None, shows the most recent data.

        assert param_name is not None, 'Parameter param_name not defined, aborting...'
        assert startswith is not None, 'Parameter startswith not defined. Use "NG" or "S", aborting...'
        data = self.getData(filename=results_filename, data_type='results')

        # Visualize
        assert f'{param_name}_all' in data.keys(),f'No {param_name} data found. Was it recorded? Aborting...'
        group_list = [n for n in data[f'{param_name}_all'].keys() if n.startswith(f'{startswith}')]
        print(group_list)
 
        t=data[f'{param_name}_all'][group_list[0]]['t']
        time_array = t / t.get_best_unit()
        fig, axs = self._prep_group_figure(group_list)

        for ax, this_group in zip(axs,group_list):

            N_monitored_neurons, this_data = self._get_n_neurons_and_data_array(data, this_group, param_name, neuron_index=neuron_index)

            if hasattr(this_data,'get_best_unit'):
                data_array = this_data / this_data.get_best_unit()
                this_unit = this_data.get_best_unit()
            else:
                data_array = this_data
                this_unit = '1'
            im = ax.plot(time_array, data_array)
            ax.set_title(this_group, fontsize=10)
            ax.set_xlabel(t.get_best_unit())
            ax.set_ylabel(this_unit)

        fig.suptitle(f'{param_name}', fontsize=16)
        
        if savefigname:
            self._figsave(figurename=savefigname)

    def show_currents(self, results_filename=None, savefigname='', neuron_index=None, t_idx_start=None, t_idx_end=None):

        data = self.getData(results_filename, data_type='results')
        # Visualize
        # Extract connections from data dict
        list_of_results_ge = [n for n in data['ge_soma_all'].keys() if 'NG' in n]
        list_of_results_gi = [n for n in data['gi_soma_all'].keys() if 'NG' in n]
        list_of_results_vm = [n for n in data['vm_all'].keys() if 'NG' in n]

        assert list_of_results_ge == list_of_results_gi == list_of_results_vm, 'Some key results missing, aborting...'
        NG_list = list_of_results_ge 

        t = data['ge_soma_all'][NG_list[0]]['t']

        if t_idx_start is None:
            t_idx_start = 0
        if t_idx_end is None:
            t_idx_end = len(t)

        fig, axs = self._prep_group_figure(NG_list)

        for ax, NG in zip(axs, NG_list):

            I_e, I_i, I_leak = self._get_currents_by_interval(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)

            if neuron_index is None:
                # print(f'Showing mean current over all neurons for group {NG}')
                N_neurons = I_e.shape[1]
                I_e_display = I_e.sum(axis=1) / N_neurons
                I_i_display = I_i.sum(axis=1) / N_neurons
            else:
                I_e_display = I_e[:,neuron_index[NG]]
                I_i_display = I_i[:,neuron_index[NG]]

            # Inhibitory current is negative, invert to cover same space as excitatory current
            I_i_display = I_i_display * -1

            ax.plot(t[t_idx_start:t_idx_end], np.array([I_e_display, I_i_display]).T)       

            ax.legend(['I_e', 'I_i'])
            ax.set_title(NG + ' current', fontsize=10)

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_connections(self, connections_filename=None, hist_from=None, savefigname=''):

        data = self.getData(filename=connections_filename, data_type='connections')

        # Visualize
        # Extract connections from data dict
        list_of_connections = [n for n in data.keys() if '__to__' in n]
    
        # Pick histogram data
        if hist_from is None:
            hist_from = list_of_connections[-1]

        print(list_of_connections)
        fig, axs = self._prep_group_figure(list_of_connections)

        for ax, connection in zip(axs,list_of_connections):
            im = ax.imshow(data[connection]['data'].todense())
            ax.set_title(connection, fontsize=10)
            fig.colorbar(im, ax=ax)
        data4hist = np.squeeze(np.asarray(data[hist_from]['data'].todense().flatten()))
        data4hist_nozeros = np.ma.masked_equal(data4hist,0)
        Nzeros = data4hist==0
        proportion_zeros = Nzeros.sum() / Nzeros.size

        n_images=len(list_of_connections)
        n_rows = int(np.ceil(n_images/2))
        axs[(n_rows * 2)-1].hist(data4hist_nozeros)
        axs[(n_rows * 2)-1].set_title(f"{hist_from}\n{(proportion_zeros * 100):.1f}% zeros (not shown)")
        if savefigname:
            self._figsave(figurename=savefigname)

    def _make_2D_surface(self, fig, ax, data, x_values=None, y_values=None, x_label=None, y_label=None, variable_name=None, variable_unit=None):

        sns.heatmap(data, cmap=self.cmap, ax=ax, cbar=True, annot=True, fmt='.3g', xticklabels=x_values, yticklabels=y_values)

        # Set common labels
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)

    def _make_3D_surface(self, ax, x_values, y_values, z_values, x_label=None, y_label=None, variable_name=None):

        X, Y = np.meshgrid(x_values, y_values)
        dense_grid_steps = 50 # This should be relative to search space density
        assert dense_grid_steps > x_values.shape[0], 'Interpolation to less than original data, aborting...' # You can comment this out and think it as a warning
        grid_x, grid_y = np.mgrid[np.min(x_values):np.max(x_values):dense_grid_steps*1j, np.min(y_values):np.max(y_values):dense_grid_steps*1j]
        values = z_values.flatten()
        points = np.array([X.flatten(), Y.flatten()]).T
        grid_z2 = griddata(points, values, (grid_x, grid_y), method='nearest') #, linear, nearest, cubic
        ax.plot_surface(grid_x, grid_y, grid_z2, ccount=50, rcount=50, cmap = self.cmap)

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if variable_name is not None:
            ax.set_zlabel(variable_name)

    def _make_table(self, ax, text_keys_list=[], text_values_list=[]):

        for i, (this_key, this_value) in enumerate(zip(text_keys_list, text_values_list)):
            # ax.text(0.01, 0.9, f"{this_key}: {this_value}", va="top", ha="left")
            j=i - (i//5) * 5
            ax.text(0.01 + (i//5) * 0.3, 0.8 - (j * 0.15), f"{this_key}: {this_value}")
            ax.tick_params(labelbottom=False, labelleft=False)

        ax.tick_params(
            axis='both',        # changes apply to both x and y axis; 'x', 'y', 'both'
            which='both',       # both major and minor ticks are affected
            left=False,         # ticks along the left edge are off
            bottom=False,       # ticks along the bottom edge are off
            labelleft=False,    # labels along the left edge are off
            labelbottom=False)  

    def _prep_group_figure(self, NG_list):

        n_images = len(NG_list)
        if n_images == 1:
            n_columns = 1
        else:
            n_columns = 2
    
        n_rows = int(np.ceil(n_images/n_columns))

        fig, axs = plt.subplots(n_rows, n_columns)
        
        return fig, fig.axes

    def _prep_array_figure(self, two_dim):

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

        if two_dim:
            ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
            ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2,projection='3d')
        else:
            ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2)

        return fig, fig.axes

    def show_analyzed_arrayrun(self, csv_filename=None, analysis=None, variable_unit=None, NG_id_list=[]):
        '''
        Pseudocode
        Get MeanFR_TIMESTAMP_.csv
        If does not exist, calculate from metadata file list
        Prep figure in subfunction, get axes handles
        Table what is necessary, display
        Plot 2D
        Plot 3D
        '''

        assert analysis.lower() in SystemAnalysis.map_analysis_names.keys(), 'Unknown analysis, aborting...'

        analysisHR = self.map_analysis_names[analysis.lower()]
        # Get MeanFR_TIMESTAMP_.csv

        try:
            if csv_filename is None:
                data_df = self.getData(data_type=analysisHR)
            else:
                data_df = self.getData(filename=csv_filename, data_type=None)

        # If does not exist, calculate from metadata file list
        except FileNotFoundError as error:
            print(error)
            print('Conducting necessary analysis first. Using most recent metadata file and full duration')
            self.analyze_arrayrun(analysis=analysisHR)
            data_df = self.getData(data_type=analysisHR)

        if analysisHR.lower() in ['meanfr', 'meanvm', 'eicurrentdiff']:
            print(f'Creating one figure for each neuron group')
        elif analysisHR.lower() in ['grcaus', 'classify']:
            print(f'Creating one figure for each analysis')

        analyses_for_zipping = [analysisHR] * len(data_df.columns)
        available_data_column_list = [  ng for (dtype, ng) in zip(analyses_for_zipping, data_df.columns) 
                                        if dtype.lower() in ng.lower()]
        if not NG_id_list:
            print('All neuron groups requested')
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
            start_idx = this_data_column.find('NG')
            end_idx = this_data_column.find('_', start_idx)
            NG_id_list.append(this_data_column[start_idx:end_idx])
            uscore_idx = this_data_column.find('_', end_idx) + 1
            NG_name_list.append(this_data_column[uscore_idx:])

        if 'Dimension-2 Parameter' in data_df.columns:
            two_dim = True
        else:
            two_dim = False

        if analysisHR.lower() in ['grcaus']: 
            variable_unit_dict = {'_p' : 'p value', '_Information' : '(bits)', '_latency' : 'latency (s)', 
            '_TransfEntropy' : 'Transfer Entropy (bits/sample)', '_isStationary' : 'boolean', 
            '_targetEntropy' : 'target signal entropy (bits)', '_fitQuality' : 'mean fit quality'}

        for this_NG_id, this_NG_name, this_data_column in zip(NG_id_list, NG_name_list, requested_data_column_list):

            assert this_NG_id in this_data_column, 'Neuron group does not match data column, aborting ...'

            if 'variable_unit_dict' in locals():
                for this_key in variable_unit_dict.keys():
                    if this_key in this_NG_name:
                        variable_unit = variable_unit_dict[this_key]

            # Prep figure in subfunction, get axes handles
            fig, axs = self._prep_array_figure(two_dim)
            
            # Table what is necessary, display
            text_keys_list=['Analysis', 'Neuron Group #', 'Neuron Group Name', 'MIN value - (y,x)', 
                'MAX value - (y,x)', 'MIN at Params', 'MAX at Params']
            text_values_list=[]
            text_values_list.append(analysisHR)
            text_values_list.append(this_NG_id)
            text_values_list.append(this_NG_name)

            value_column_name = this_data_column
            variable_name=analysisHR
            
            if two_dim:
                # Get 2 dims for viz
                index_column_name = 'Dimension-1 Value'
                column_column_name = 'Dimension-2 Value'
                x_label=data_df['Dimension-2 Parameter'][0]
                y_label=data_df['Dimension-1 Parameter'][0]

                df_2d = self.pivot_to_2d_dataframe(data_df, index_column_name=index_column_name, 
                    column_column_name=column_column_name, value_column_name=value_column_name)
                data_nd_array = df_2d.values
                x_values = df_2d.columns
                y_values = df_2d.index
            else:
                x_label = data_df['Dimension-1 Parameter'][0]
                data_nd_array = data_df[value_column_name].values
                x_values = data_df["Dimension-1 Value"].values

            min_value = np.amin(data_nd_array)
            min_value_rounded = self.round_to_n_significant(min_value, significant_digits=3)
            min_idx = np.unravel_index(np.argmin(data_nd_array), data_nd_array.shape)
            max_value = np.amax(data_nd_array)
            max_value_rounded = self.round_to_n_significant(max_value, significant_digits=3)
            max_idx = np.unravel_index(np.argmax(data_nd_array), data_nd_array.shape)
            text_values_list.append(f'{min_value_rounded} {variable_unit}- {min_idx}')
            text_values_list.append(f'{max_value_rounded} {variable_unit} - {max_idx}')

            if two_dim:
                text_values_list.append(f'{y_label} = {df_2d.index[min_idx[0]]}; {x_label} = {df_2d.columns[min_idx[1]]}')
                text_values_list.append(f'{y_label} = {df_2d.index[max_idx[0]]}; {x_label} = {df_2d.columns[max_idx[1]]}')
            else:
                text_values_list.append(f'{x_label} = {data_df["Dimension-1 Value"][min_idx[0]]}')
                text_values_list.append(f'{x_label} = {data_df["Dimension-1 Value"][max_idx[0]]}')
                
            self._make_table(axs[0], text_keys_list=text_keys_list, text_values_list=text_values_list)

            if two_dim:
                self. _make_2D_surface(fig, axs[1], data_nd_array, x_values=x_values, y_values=y_values, 
                    x_label=x_label, y_label=y_label, variable_name=variable_name, variable_unit=variable_unit)

                self._make_3D_surface(axs[2], x_values, y_values, data_nd_array, 
                    x_label=x_label, y_label=y_label, variable_name=variable_name)
            else:
                axs[1].plot(x_values, data_nd_array)
                axs[1].set_xlabel(f'{x_label}' )
                axs[1].set_ylabel(f'{variable_name} ({variable_unit})' )

            if  hasattr(self,'save_figure_with_arrayidentifier'):
                if analysisHR.lower() in ['meanfr', 'meanvm', 'eicurrentdiff']:
                    identifier = this_NG_id
                else:
                    # Assuming the last word in this_NG_name contains the necessary identifier
                    suffix_start_idx = this_NG_name.rfind('_')
                    identifier = this_NG_name[suffix_start_idx + 1:]
                
                self._figsave(figurename=f'{self.save_figure_with_arrayidentifier}_{analysisHR}_{identifier}', myformat='svg')

    def show_input_to_readout_coherence(self, results_filename=None, savefigname='', signal_pair=[0,0]):

        data_dict = self.getData(filename=results_filename, data_type='results')

        analog_input = self.getData( self.input_filename, data_type=None)
        source_signal = analog_input['stimulus'].T # We want time x units


        NG = [n for n in data_dict['vm_all'].keys() if 'NG3' in n]
        # vm_unit = self._get_vm_by_interval(data_dict, NG[0], t_idx_start=0, t_idx_end=-1)
        vm_unit = self._get_vm_by_interval(data_dict, NG[0])

        target_signal = vm_unit / vm_unit.get_best_unit() 

        cut_length = 100
        x = source_signal[cut_length:-cut_length,signal_pair[0]]
        y = target_signal[cut_length:-cut_length,signal_pair[1]]

        high_cutoff = 100 # Frequency in Hz
        nsamples = self._get_nsamples(data_dict) # self._get_nsamples(data_dict) // downsampling_factor
        nperseg = nsamples//6 
        dt = self._get_dt(data_dict)
        samp_freq = 1.0 / dt # 1.0/(dt * downsampling_factor) 

        f, Cxy, Pwelch_spec_x, Pwelch_spec_y, Pxy, lags, corr, coherence_sum, x_scaled, y_scaled = \
            self.get_coherence_of_two_signals(x, y, samp_freq=samp_freq, nperseg=nperseg, high_cutoff=high_cutoff)
        shift_in_seconds = self._get_cross_corr_latency(lags, corr, dt)
        self._show_coherence_of_two_signals(f, Cxy, Pwelch_spec_x, Pwelch_spec_y, Pxy, lags, corr, \
            shift_in_seconds, x, y, x_scaled, y_scaled)

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_estimate_on_input( self, results_filename=None, simulation_engine='cxsystem', readout_group='E', 
                                decoding_method='least_squares', output_type='estimated', unit_idx_list=[0]):

        Error, xL, xest = self.get_MSE( results_filename=results_filename, simulation_engine=simulation_engine, 
                                        readout_group=readout_group, decoding_method=decoding_method, output_type=output_type) 
        fig, ax = plt.subplots(nrows=1,ncols=1)
        ax.plot(xL[:,unit_idx_list])
        ax.plot(xest[:,unit_idx_list])

        self._string_on_plot(ax, variable_name='Error', variable_value=Error, variable_unit='a.u.')
        
        if len(unit_idx_list) == 1:
            plt.legend(['Target', 'Estimate'])

    def get_system_profile_metrics(self, data_df_compiled, independent_variable_columns_list):

        # Turn array analysis columns into system profile metrics

        def _column_comprehension(analysis, key_list=[]):

            relevant_columns = [col for col in data_df_compiled.columns if f'{analysis}' in col]
            pruned_names = []
            if key_list is not []:
                key_columns = []
                for this_key in key_list:
                    this_key_column = [col for col in relevant_columns if f'{this_key}' in col]
                    key_columns.extend(this_key_column)
                    pruned_names.append(f'{analysis}_{this_key}')
                relevant_columns = key_columns
            if pruned_names is []:
                    pruned_names = [f'{analysis}']
            return relevant_columns, pruned_names
        
        profile_metrics_columns_list = []

        # Init df_for_barplot with independent columns
        df_for_barplot = data_df_compiled[independent_variable_columns_list]

        ## Get energy metrics ## 
        # From Attwell_2001_JCerBlFlMetab.pdf "As a simplification, all cells are treated as glutamatergic, 
        # because excitatory neurons outnumber inhibitory cells by a factor of 9 to 1, and 90% of synapses 
        # release glutamate (Abeles, 1991; Braitenberg and Schüz, 1998)."

        meanfr_column_list, pruned_names = _column_comprehension('MeanFR', key_list=['NG1'])
        selected_columns = data_df_compiled[meanfr_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        ## Get latency metrics ##
        latency_column_list, pruned_names = _column_comprehension('Coherence', key_list=['Latency'])
        selected_columns = data_df_compiled[latency_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        # latency_column_list, pruned_names = _column_comprehension('GrCaus', key_list=['latency'])
        # selected_columns = data_df_compiled[latency_column_list]
        # df_for_barplot[pruned_names] = selected_columns
        # profile_metrics_columns_list.extend(pruned_names)

        ## Get reconstruction metrics ##
        reco_column_list, pruned_names = _column_comprehension('MeanError', key_list=['SimErr'])
        selected_columns = data_df_compiled[reco_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        # ## Get classification accuracy ## 
        # classify_column_list, pruned_names = _column_comprehension('Classify', key_list=['Accuracy'])
        # selected_columns = data_df_compiled[classify_column_list]
        # df_for_barplot[pruned_names] = selected_columns
        # profile_metrics_columns_list.extend(pruned_names)

        ## Get information metrics ##
        info_column_list, pruned_names = _column_comprehension('GrCaus', key_list=['Information'])
        selected_columns = data_df_compiled[info_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        info_column_list, pruned_names = _column_comprehension('GrCaus', key_list=['TransfEntropy'])
        selected_columns = data_df_compiled[info_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        ## Get target entropy ##
        target_entropy_column_list, pruned_names = _column_comprehension('GrCaus', key_list=['targetEntropy'])
        selected_columns = data_df_compiled[target_entropy_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        ## Get input-output coherence ##
        coherence_column_list, pruned_names = _column_comprehension('Coherence', key_list=['Sum'])
        selected_columns = data_df_compiled[coherence_column_list]
        df_for_barplot[pruned_names] = selected_columns
        profile_metrics_columns_list.extend(pruned_names)

        # TÄHÄN JÄIT: MUUTA LATENCY=>SPEED ETC, ELI SUUREMPI PAREMPI (?) JA JATKA PLOTTIIN
        # JAA COHERENSSI ALLE JA YLI 25 Hz. pALAUTA gRCAUS, TEE VAR BIC KRITEERISTÄ NÄKYVÄ

        min_values = df_for_barplot[profile_metrics_columns_list].min()
        max_values = df_for_barplot[profile_metrics_columns_list].max()
        
        return df_for_barplot, profile_metrics_columns_list, min_values, max_values

    def PC2RGB(self, cardinal_points, additional_points):
        # Get transformation matrix between 2D cardinal points as corners of a rectangular space 
        # [left lower, right upper, left upper, right lower]. Map this space to RGB gamut in CIExy color space.
        # Transform additional points between the spaces
        # returns the additional points 

        # Major color axis contained in RGB gamut in CIE xy coordinates
        CIE_cardinal_points_list = [[0.6, 0.35], [0.3, 0.5], [0.2, 0.15], [0.45, 0.45]]
        CIE_cardinal_points_df = pd.DataFrame(CIE_cardinal_points_list, index=['red', 'green', 'blue', 'yellow'], columns=['CIE_x', 'CIE_y'])
        CIE_matrix = CIE_cardinal_points_df.values

        # Get transformation matrix X between PC dimensions and CIExy coordinates 
        PC_matrix_ones = np.c_[cardinal_points,np.ones([4,1])]

        CIE_matrix_ones = np.c_[CIE_matrix,np.ones([4,1])]
        transf_matrix = np.linalg.lstsq(PC_matrix_ones, CIE_matrix_ones)[0]

        CIE_space = np.c_[additional_points, np.ones(additional_points.shape[0])] @ transf_matrix
        CIEonXYZ = colour.xy_to_XYZ(CIE_space[:,:2])
        CIEonRGB = colour.XYZ_to_sRGB(CIEonXYZ / 100)

        # Get standard scaler
        CIE_standard_space = PC_matrix_ones @ transf_matrix
        CIEonXYZ_standard_space = colour.xy_to_XYZ(CIE_standard_space[:,:2])
        CIEonRGB_standard_space = colour.XYZ_to_sRGB(CIEonXYZ_standard_space / 100)
        standard_min = np.min(CIEonRGB_standard_space)
        standard_ptp = np.ptp(CIEonRGB_standard_space)

        # Scale brighter
        # RGB_points = (CIEonRGB - np.min(CIEonRGB)) / np.ptp(CIEonRGB)
        RGB_points_scaled = (CIEonRGB - standard_min) / standard_ptp
        # clip to [0, 1]
        RGB_points = np.clip(RGB_points_scaled, 0, 1)

        return RGB_points

    def system_polar_bar(self, row_selection = None, folder_name=None):

        if isinstance(row_selection, list) and len(row_selection)>1:
            # For list of rows, make one figure for each row
            for this_row in row_selection:
                self.system_polar_bar(row_selection = this_row, folder_name=folder_name)
            return
        elif isinstance(row_selection, list) and len(row_selection)==1:
            row_selection = row_selection[0]
        
        ## Get csv files recursively from folder and its subfolders ##
        if folder_name is None:
            out_root_path = os.path.join(self.path, self.output_folder)
        else:
            out_root_path = os.path.join(self.path, folder_name)

        root_subs_files_list = [tp for tp in os.walk(out_root_path)]
        csv_file_list = []
        for this_tuple in root_subs_files_list:
            this_root_path = this_tuple[0]
            this_csv_file_list = [p for p in this_tuple[2] if p.endswith('.csv')]
            for this_file in this_csv_file_list:
                first_underscore_idx = this_file.find('_')
                if this_file[:first_underscore_idx] not in self.map_analysis_names.values():
                    raise ValueError(f'{this_file} is unknown csv file, allowed types should start with {self.map_analysis_names.values()}')
                this_csv_file_path = os.path.join(this_root_path, this_file)
                csv_file_list.append(this_csv_file_path)

        ## Get data from csv files ##
        data0_df = self.getData(filename=csv_file_list[0], data_type=None)
        if "Dimension-2 Parameter" in data0_df.columns:
            independent_variable_columns_list = ["Dimension-1 Parameter","Dimension-1 Value", "Dimension-2 Parameter", "Dimension-2 Value"]
        else: 
            independent_variable_columns_list = ["Dimension-1 Parameter","Dimension-1 Value"]
        selected_columns = data0_df[independent_variable_columns_list]
        data_df_compiled = selected_columns.copy()
        dependent_variable_columns_list = []

        for csv_filename in csv_file_list:
            data_df = self.getData(filename=csv_filename, data_type=None)
            # Drop 'Unnamed: 0'
            if 'Unnamed: 0' in data_df.columns:
                data_df = data_df.drop(['Unnamed: 0'], axis=1)
            # If independent dimensions match, add column values
            if data_df[independent_variable_columns_list].equals(data_df_compiled[independent_variable_columns_list]):
                # Get list of dependent variable columns
                this_dependent_variable_columns_list = [col for col in data_df.columns if col not in independent_variable_columns_list]
                data_df_compiled[this_dependent_variable_columns_list] = data_df[this_dependent_variable_columns_list]
                dependent_variable_columns_list.extend(this_dependent_variable_columns_list)

        # Combine dfs
        df_for_barplot, profile_metrics_columns_list, min_values, max_values = \
            self.get_system_profile_metrics(data_df_compiled, independent_variable_columns_list)

        # Normalize magnitudes
        values_np = df_for_barplot[profile_metrics_columns_list].values #returns a numpy array
        values_np_scaled = self.scaler(values_np, scale_type='minmax', feature_range=[0, 1])
        df_for_barplot[profile_metrics_columns_list] = values_np_scaled

        # extra points are the original dimensions, to be visualized
        extra_points = np.vstack([np.eye(len(profile_metrics_columns_list)), -1 * np.eye(len(profile_metrics_columns_list))])

        # Get PCA of data. Note option for extra_points=extra_points_df
        values_pca, principal_axes_in_PC_space, explained_variance_ratio, extra_points_pca = self.get_PCA(values_np, 
            n_components=2, col_names=profile_metrics_columns_list, extra_points=extra_points,
            extra_points_at_edge_of_gamut=True)

        # Define PCA space         
        xmin, xmax = np.min(values_pca[:,0]), np.max(values_pca[:,0])
        ymin, ymax = np.min(values_pca[:,1]), np.max(values_pca[:,1])
        PC0_limits = np.array([xmin - xmin/10, xmax + xmax/10])
        PC1_limits = np.array([ymin - ymin/10, ymax + ymax/10])

        PC_cardinal_points = np.array([  [PC0_limits[0],PC1_limits[0]],
                                [PC0_limits[1],PC1_limits[1]],
                                [PC0_limits[0],PC1_limits[1]],
                                [PC0_limits[1],PC1_limits[0]]]) 

        # Number of parameters for the polar bar chart
        N = len(profile_metrics_columns_list)

        # Polar angle (in radians) of each parameter
        theta = np.linspace(2 * np.pi / N, 2 * np.pi, N, endpoint=True)

        # Magnitude as radius of bar in the polar plot
        if row_selection is None:
            row_selection = 0
        radii = df_for_barplot.iloc[row_selection,:][profile_metrics_columns_list].values

        # Second dimension as width
        width = 2 * np.pi / N

        # Third dimension as colors
        profile_in_PC_coordinates = extra_points_pca
        profile_in_RGB_coordinates = self.PC2RGB(PC_cardinal_points, additional_points = profile_in_PC_coordinates)

        # colors = plt.cm.viridis(radii.astype(float))
        colors = profile_in_RGB_coordinates

        ## Plotting##
        fig = plt.figure(figsize=(10,5))
        ax1 = plt.subplot(221, projection='polar')
        ax2 = plt.subplot(222, projection=None)
        ax3 = plt.subplot(223, projection=None)
        ax4 = plt.subplot(224, projection=None)

        ax1.bar(theta, radii, width=width, bottom=0.0, color=colors)
        
        # Polar tick positions in radial coordinates and corresponding label strings
        ax1.set_xticks(np.linspace(2*np.pi/(N),2*np.pi,N))
        ax1.set_xticklabels(profile_metrics_columns_list)

        # Set radial max value
        ax1.set_rmax(1.0)
        # Radial ticks
        ax1.set_rticks([0.25, 0.5, 0.75, 1.0])  # Set radial ticks
        ax1.set_rlabel_position(-11.25)  # Move radial labels away from plotted line

        # Subplot title; selected point
        parameter1name = f'{data_df_compiled.loc[row_selection, independent_variable_columns_list[0]]}'
        parameter1value = f'{data_df_compiled.loc[row_selection, independent_variable_columns_list[1]]:.2f}'
        if "Dimension-2 Parameter" in data0_df.columns:
            parameter2name = f'{data_df_compiled.loc[row_selection, independent_variable_columns_list[2]]}'
            parameter2value = f'{data_df_compiled.loc[row_selection, independent_variable_columns_list[3]]:.2f}'
            ax1.title.set_text(f'{parameter1name}: {parameter1value}\n{parameter2name}: {parameter2value}')
            figout_names = parameter1name.replace(' ','_') + parameter1value.replace('.','p') + '_' \
                + parameter2name.replace(' ','_') + parameter2value.replace('.','p')
        else:
            ax1.title.set_text(f'{parameter1name}: {parameter1value}')
            figout_names = parameter1name.replace(' ','_') + parameter1value.replace('.','p')


        # Legend in another subplot (too big for the same)
        min_values = min_values.apply(self.round_to_n_significant, args=(3,)).values
        max_values = max_values.apply(self.round_to_n_significant, args=(3,)).values
        bar_dict = {f'{name}: min {minn}, max {maxx}': color for name, minn, maxx, color in zip(profile_metrics_columns_list, min_values, max_values, colors)}         
        labels = list(bar_dict.keys())
        handles = [plt.Rectangle((0,0),1,1, color=bar_dict[label]) for label in labels]
        ax2.set_axis_off() 
        ax2.legend(handles, labels, loc='center')

        # Plot data on PC space with colors from CIExy map
        RGB_values = self.PC2RGB(PC_cardinal_points, additional_points = values_pca)
        ax3.scatter(values_pca[:,0], values_pca[:,1], c=RGB_values)

        # Selected point
        ax3.scatter(values_pca[row_selection,0], values_pca[row_selection,1], c=RGB_values[row_selection], edgecolors='k', s=100)

        # Original N dim space, projected on 2 dim PC space
        extra_point_markers = "s"
        n_metrics = len(profile_metrics_columns_list)

        # End points
        ax3.scatter(extra_points_pca[:n_metrics,0], extra_points_pca[:n_metrics,1], marker = extra_point_markers, c='k')
        pc0_label = f'PC0 ({100*explained_variance_ratio[0]:.0f}%)'
        pc1_label = f'PC1 ({100*explained_variance_ratio[1]:.0f}%)'
        ax3.set_xlabel(pc0_label)
        ax3.set_ylabel(pc1_label)
        ax3.grid()

        ## Display colorscale
        x = np.linspace(PC0_limits[0], PC0_limits[1], num=100)
        y = np.linspace(PC1_limits[0], PC1_limits[1], num=100)
        X,Y = np.meshgrid(x,y)
        PC_space = np.vstack([X.flatten(),Y.flatten()]).T       
        CIEonRGB_scaled = self.PC2RGB(PC_cardinal_points, additional_points = PC_space)

        # Reshape to 2D
        imsize = (len(X),len(Y),3)
        CIEonRGB_image = np.reshape(CIEonRGB_scaled, imsize, order='C')
        # Get extent of existing 2 dim PC plot
        PC_ax_extent = [ax3.get_xlim(), ax3.get_ylim()]
        # Plot colorscape
        ax4.imshow(CIEonRGB_image, origin='lower', extent=[PC_ax_extent[0][0], PC_ax_extent[0][1], PC_ax_extent[1][0], PC_ax_extent[1][1]])
        ax4.grid()

        # Create plot pairs
        for this_point in np.arange(n_metrics):
            start_end_x = [extra_points_pca[this_point,0], extra_points_pca[n_metrics + this_point,0]]
            start_end_y = [extra_points_pca[this_point,1], extra_points_pca[n_metrics + this_point,1]]
            ax4.plot(start_end_x, start_end_y, 'k--', lw=.5)
        # # Start points
        # ax4.scatter(extra_points_pca[n_metrics:,0], extra_points_pca[n_metrics:,1], marker = extra_point_markers, c='w', edgecolors='k', s=20)        
        # End points
        ax4.scatter(extra_points_pca[:n_metrics,0], extra_points_pca[:n_metrics,1], marker=extra_point_markers, facecolors='none', edgecolors='k')
        ax4.title.set_text(f'Projections of the {str(len(profile_metrics_columns_list))} dimensions')


        if  hasattr(self,'save_figure_with_arrayidentifier'):                
            self._figsave(figurename=f'{self.save_figure_with_arrayidentifier}_summary_{figout_names}', myformat='svg')

if __name__=='__main__':

    # path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'

    # SV = SystemViz(path=path)

    # # analysis.printMeanFR(filename=None, t_idx_start=0, t_idx_end=None)

    # # Neuron group names 'NG1_L4_SS_L4', 'NG2_L4_BC_L4', 'NG3_L4_SS2_L4'
    # NG_name = 'NG3_L4_SS2_L4'
    # df = SV.show_readout_on_input(NG_name, normalize=True)

    # # df_long = SV.unpivot_dataframe(df, index_column=['t'], kw_sub_to_unpivot='MeanFR')
    # sns.lineplot(   x="t", y='data', hue='units',
    #                 data=df)

    # # SV = SystemViz()

    # # # path = r'/home/tgarnier/CxPytestWorkspace/matrixsearch_L4_BC_noise'
    # # path = r'C:\Users\Simo\Laskenta\SimuOut'
    # # metadata_filename = 'MeanFR__20201203_2029581.csv'
    # # metadata_fullpath = os.path.join(path,metadata_filename)
    
    # # df = pd.read_csv(metadata_fullpath, index_col=0)
    # # index_column = ['Dimension-1 Value', 'Dimension-2 Value']
    # # df_long = SV.unpivot_dataframe(df, index_column=index_column, kw_sub_to_unpivot='MeanFR')
    # # print(df_long)

    # # g = sns.FacetGrid(df_long, col="groupName", col_wrap=2, height=2)    
    # # g.map(sns.pointplot, "Dimension-1 Value", "MeanFR")

    # # g2 = sns.FacetGrid(df_long, col="groupName", col_wrap=2, height=2)    
    # # g2.map(sns.pointplot, "Dimension-2 Value", "MeanFR")
    # # groups=['MeanFR_NG0_relay_spikes', 'MeanFR_NG1_L4_SS_L4',
    # #    'MeanFR_NG2_L4_BC_L4', 'MeanFR_NG3_L4_PC1_L4toL1']
    # # group_0 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[0])
    # # group_1 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[1])
    # # group_2 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[2])
    # # group_3 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[3])

    # # plt.figure()
    # # ax = sns.heatmap(group_0)
    # # plt.title(groups[0])

    # # plt.figure()
    # # ax = sns.heatmap(group_1)
    # # plt.title(groups[1])

    # # plt.figure()
    # # ax = sns.heatmap(group_2)
    # # plt.title(groups[2])
    
    # # plt.figure()
    # # ax = sns.heatmap(group_3)
    # # plt.title(groups[3])

    # plt.show()
    pass