# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


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
        for column_name in df.columns:
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

    def plot_readout_on_input(self, results_filename=None, normalize=False, unit_idx_list=None):
        '''
        Get input, get data. Scaling. turn to df, format df, Plot curves.
        '''
        # Get data and input
        data = self.getData(filename=results_filename, data_type='results')

        analog_input = self.getData( self.input_filename, data_type=None)

        analog_signal = analog_input['stimulus'].T
        assert analog_signal.ndim == 2, 'input is not a 2-dim vector, aborting...'
        # analog_timestep = analog_input['frameduration']

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
        # plt.show()

    def lineplot(self, data):
        data_df = pd.DataFrame(data)
        sns.lineplot(data=data_df)
        plt.show()

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

    def show_analog_results(self, results_filename=None, savefigname='',param_name=None,startswith=None):
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

        for ax, results in zip(axs,group_list):
            N_monitored_neurons = data[f'{param_name}_all'][results][f'{param_name}'].shape[1]
            this_data =  data[f'{param_name}_all'][results][f'{param_name}']
            if hasattr(this_data,'get_best_unit'):
                data_array = this_data / this_data.get_best_unit()
                this_unit = this_data.get_best_unit()
            else:
                data_array = this_data
                this_unit = '1'
            im = ax.plot(time_array, data_array)
            ax.set_title(results, fontsize=10)
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

        im = ax.imshow( data, cmap=self.cmap, interpolation='none', 
                        extent=[np.min(x_values),np.max(x_values),np.max(y_values),np.min(y_values)])
        pos1 = ax.get_position() # get the original position    

        left =  pos1.xmax
        bottom = pos1.ymin
        width = (pos1.xmax - pos1.xmin)/10
        height = pos1.ymax - pos1.ymin
        cax = fig.add_axes([left, bottom, width, height])
        fig.colorbar(im, cax=cax, orientation='vertical')
        plot_str = f'{variable_name} {variable_unit}'
        cax.set_ylabel(plot_str)

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
        grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
        ax.plot_surface(grid_x, grid_y, grid_z2, cstride=1, rstride=1, cmap = self.cmap)

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

        print(f'Creating one figure for each neuron group')
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
            variable_unit_dict = {'_p' : 'p value', '_logF' : 'log2(F)', '_latency' : 'latency (s)', 
            '_isStationary' : 'boolean', '_target_entropy' : 'entropy (bits)', '_fit_quality' : 'mean fit quality'}

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
            min_value_rounded = self.round_to_n_significant(min_value, significant_digits=2)
            min_idx = np.unravel_index(np.argmin(data_nd_array), data_nd_array.shape)
            max_value = np.amax(data_nd_array)
            max_value_rounded = self.round_to_n_significant(max_value, significant_digits=2)
            max_idx = np.unravel_index(np.argmax(data_nd_array), data_nd_array.shape)
            text_values_list.append(f'{min_value_rounded} {variable_unit}- {min_idx}')
            text_values_list.append(f'{max_value_rounded} {variable_unit} - {max_idx}')

            if two_dim:
                text_values_list.append(f'{y_label} = {df_2d.columns[min_idx[0]]}; {x_label} = {df_2d.index[min_idx[1]]}')
                text_values_list.append(f'{y_label} = {df_2d.columns[max_idx[0]]}; {x_label} = {df_2d.index[max_idx[1]]}')
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

if __name__=='__main__':

    # path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'

    # SV = SystemViz(path=path)

    # # analysis.printMeanFR(filename=None, t_idx_start=0, t_idx_end=None)

    # # Neuron group names 'NG1_L4_SS_L4', 'NG2_L4_BC_L4', 'NG3_L4_SS2_L4'
    # NG_name = 'NG3_L4_SS2_L4'
    # df = SV.plot_readout_on_input(NG_name, normalize=True)

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