# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Analysis
import numpy as np
import pandas as pd

# Builtin
import os

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

        # columns for input
        new_data_shape = np.asarray(new_data.shape)
        new_data_time_dim = np.where(new_data_shape == Ntimepoints)
        if new_data_time_dim[0] == 1:
            new_data = new_data.T
            new_data_shape = np.asarray(new_data.shape)

        for ch_idx in np.arange(new_data_shape[1]):
            ch_name = f'{datatype}_ch_{ch_idx}'
            ch_data = new_data[:,ch_idx]
            data_dict[ch_name] = ch_data

        dims = new_data_shape[1]
        return data_dict, dims

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

        if normalize==True:
            V_res = data['Neuron_Groups_Parameters'][NG_name]['namespace']['V_res']
            VT = data['Neuron_Groups_Parameters'][NG_name]['namespace']['VT']
            Vcut = data['Neuron_Groups_Parameters'][NG_name]['namespace']['Vcut']
            # Check that values of reset, threshold and vm are reasonable
            reasonable = np.array([-100,0]) * b2u.mvolt

            assert all( [(reasonable[0] < V_res) & (V_res < reasonable[1]), \
                        (reasonable[0] < VT) & (VT < reasonable[1]), \
                        (reasonable[0] < np.min(data_vm)) & (np.max(data_vm)< Vcut)]), \
                        'Assumption about reset, threshold and vm values does not hold, aborting'

            data_vm[data_vm > VT] = VT 
            data_vm = data_vm / b2u.mvolt # strip units
            V_res = V_res / b2u.mvolt
            VT = VT / b2u.mvolt
            data_vm = (data_vm - V_res ) / np.ptp(np.array([V_res, VT]))
            analog_signal = (analog_signal - np.min(analog_signal)) / np.ptp(analog_signal)

        # Create dict and column for timepoints
        id_var = 't'
        data_dict = {id_var:t}
        Ntimepoints = t.shape[0]

        # columns for input
        data_dict, dims_IN = self._build_columns(data_dict, analog_signal, Ntimepoints, 'IN')

        # columns for vm
        data_dict, dims_Dec_vm = self._build_columns(data_dict, data_vm, Ntimepoints, 'Dec_vm')

        # Get final output dimension, to get values for unpivot
        prod_dims = dims_IN * dims_Dec_vm

        df_from_arr = pd.DataFrame(data=data_dict)

        value_vars = df_from_arr.columns[df_from_arr.columns != id_var]
        df_from_arr_unpivot = pd.melt(  df_from_arr, 
                                        id_vars=[id_var], 
                                        value_vars=value_vars, 
                                        var_name='units', 
                                        value_name='data')

        # return df_from_arr_unpivot

        sns.lineplot(   x="t", y='data', hue='units',
                data=df_from_arr_unpivot)

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
    # pdb.set_trace()
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