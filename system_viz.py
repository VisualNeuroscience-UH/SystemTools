# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


# Analysis
import numpy as np
import pandas as pd

# Builtin
import os
import copy

# Computational neuroscience
import brian2.units as b2u

# Current repo
from system_utilities import SystemUtilities

# Develop
import pdb

'''
Module on visualization

Developed by Simo Vanni 2020-2021
'''

class SystemViz(SystemUtilities):

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

    def plot_readout_on_input(self, filename=None, normalize=False):
        '''
        Get input, get data. Scaling. turn to df, format df, Plot curves.
        '''
        # Get data and input
        data = self.getData(filename, data_type='results')

        analog_input = self.getData( self.input_filename, data_type=None)

        analog_signal = analog_input['stimulus']
        assert analog_signal.ndim == 2, 'input is not a 2-dim vector, aborting...'
        analog_timestep = analog_input['frameduration']

        NG_name = self.NG_name
        t = data['vm_all'][NG_name]['t'] # All timesteps
        assert t.ndim == 1, 'timepoints are not a 1-dim vector, aborting...'

        data_vm = data['vm_all'][NG_name]['vm']

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

        # return df_from_arr_unpivot

        # palette1 = sns.color_palette("mako_r", 6)
        sns.lineplot(x="t", y='data_in', data=df_from_arr_unpivot_in, hue='units_in', palette = 'dark')
        plt.legend(loc='upper left')
        ax2 = plt.twinx()
        # palette2 = sns.color_palette("mako_r", 6)
        sns.lineplot(x="t", y='data_vm', data=df_from_arr_unpivot_vm, palette = 'bright', hue='units_vm', ax=ax2)
        plt.legend(loc='upper right')
        if normalize==True:
            EL = data['Neuron_Groups_Parameters'][NG_name]['namespace']['EL'] # data['Neuron_Groups_Parameters'][NG_name]['namespace']['EL']
            VT = data['Neuron_Groups_Parameters'][NG_name]['namespace']['VT']
            plt.ylim(EL, VT)
        # plt.show()

    def lineplot(self, data):
        data_df = pd.DataFrame(data)
        sns.lineplot(data=data_df)
        plt.show()

    def show_spikes(self, filename=None, savefigname=''):

        data = self.getData(filename, data_type='results')

        # Visualize
        # Extract connections from data dict
        NG_list = [n for n in data['spikes_all'].keys() if 'NG' in n]

        print(NG_list)
        n_images=len(NG_list)
        n_columns = 2
        n_rows = int(np.ceil(n_images/n_columns))

        fig, axs = plt.subplots(n_rows, n_columns)
        axs = axs.flat

        for ax, this_group in zip(axs,NG_list):

            im = ax.plot(data['spikes_all'][this_group]['t'], data['spikes_all'][this_group]['i'],'.')
            ax.set_title(this_group, fontsize=10)

            MeanFR = self._get_MeanFR(data, this_group, time_start=0, time_end=None)
            self._string_on_plot(ax, variable_name='Mean FR', variable_value=MeanFR, variable_unit='Hz')

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_vm(self, filename=None, savefigname=''):
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

    def show_currents(self, filename=None, savefigname='', neuron_index=None):

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

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_connections(self, filename=None, hist_from=None, savefigname=''):

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

    def _get_3D_surface(self, ax, x_values, y_values, z_values,):
        def fun(x, y):
           return x**2 + y

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-3.0, 3.0, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(fun(np.ravel(X), np.ravel(Y)))
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

if __name__=='__main__':

    path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'

    SV = SystemViz(path=path)

    # analysis.printMeanFR(filename=None, time_start=0, time_end=None)

    # Neuron group names 'NG1_L4_SS_L4', 'NG2_L4_BC_L4', 'NG3_L4_SS2_L4'
    NG_name = 'NG3_L4_SS2_L4'
    df = SV.plot_readout_on_input(NG_name, normalize=True)

    # df_long = SV.unpivot_dataframe(df, index_column=['t'], kw_sub_to_unpivot='MeanFR')
    sns.lineplot(   x="t", y='data', hue='units',
                    data=df)

    # SV = SystemViz()

    # # path = r'/home/tgarnier/CxPytestWorkspace/matrixsearch_L4_BC_noise'
    # path = r'C:\Users\Simo\Laskenta\SimuOut'
    # metadata_filename = 'MeanFR__20201203_2029581.csv'
    # metadata_fullpath = os.path.join(path,metadata_filename)
    
    # df = pd.read_csv(metadata_fullpath, index_col=0)
    # index_column = ['Dimension-1 Value', 'Dimension-2 Value']
    # df_long = SV.unpivot_dataframe(df, index_column=index_column, kw_sub_to_unpivot='MeanFR')
    # print(df_long)

    # g = sns.FacetGrid(df_long, col="groupName", col_wrap=2, height=2)    
    # g.map(sns.pointplot, "Dimension-1 Value", "MeanFR")

    # g2 = sns.FacetGrid(df_long, col="groupName", col_wrap=2, height=2)    
    # g2.map(sns.pointplot, "Dimension-2 Value", "MeanFR")
    # groups=['MeanFR_NG0_relay_spikes', 'MeanFR_NG1_L4_SS_L4',
    #    'MeanFR_NG2_L4_BC_L4', 'MeanFR_NG3_L4_PC1_L4toL1']
    # group_0 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[0])
    # group_1 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[1])
    # group_2 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[2])
    # group_3 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[3])

    # plt.figure()
    # ax = sns.heatmap(group_0)
    # plt.title(groups[0])

    # plt.figure()
    # ax = sns.heatmap(group_1)
    # plt.title(groups[1])

    # plt.figure()
    # ax = sns.heatmap(group_2)
    # plt.title(groups[2])
    
    # plt.figure()
    # ax = sns.heatmap(group_3)
    # plt.title(groups[3])

    plt.show()