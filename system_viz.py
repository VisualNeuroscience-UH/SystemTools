# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

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

    def plot_readout_on_input(self, NG_name, filename=None, filename_stimulus=None, normalize=False):
        '''
        Get input, get data. Scaling. turn to df, format df, Plot curves.
        '''
        # Get data and input
        data = self.getData(filename, data_type='results')
        # analog_input = self.getData(filename, data_type='input')
        analog_input = self.getData(filename_stimulus, data_type='input')

        analog_signal = analog_input['stimulus']
        assert analog_signal.ndim == 2, 'input is not a 2-dim vector, aborting...'
        analog_timestep = analog_input['frameduration']

        t = data['vm_all'][NG_name]['t'] # All timesteps
        assert t.ndim == 1, 'timepoints are not a 1-dim vector, aborting...'

        data_vm = data['vm_all'][NG_name]['vm']

        # if normalize==True:
        #     EL = data['Neuron_Groups_Parameters'][NG_name]['namespace']['EL'] # data['Neuron_Groups_Parameters'][NG_name]['namespace']['EL']
        #     VT = data['Neuron_Groups_Parameters'][NG_name]['namespace']['VT']
        #     Vcut = data['Neuron_Groups_Parameters'][NG_name]['namespace']['Vcut']
        #     # Check that values of reset, threshold and vm are reasonable
        #     reasonable = np.array([-100,0]) * b2u.mvolt

        #     assert all( [(reasonable[0] < EL) & (EL < reasonable[1]), \
        #                 (reasonable[0] < VT) & (VT < reasonable[1]), \
        #                 (reasonable[0] < np.min(data_vm)) & (np.max(data_vm)< Vcut)]), \
        #                 'Assumption about reset, threshold and vm values does not hold, aborting'

        #     data_vm[data_vm > VT] = VT 
        #     data_vm = data_vm / b2u.mvolt # strip units
        #     EL = EL / b2u.mvolt
        #     VT = VT / b2u.mvolt
        #     data_vm = (data_vm - EL ) / np.ptp(np.array([EL, VT]))
        #     analog_signal = (analog_signal - np.min(analog_signal)) / np.ptp(analog_signal)

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