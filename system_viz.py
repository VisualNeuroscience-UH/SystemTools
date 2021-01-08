import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import pdb

'''
Module on visualization

Developed by Simo Vanni 2020-2021
'''
class SystemViz():
    pass

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
        

if __name__=='__main__':

    SV = SystemViz()

    # path = r'/home/tgarnier/CxPytestWorkspace/matrixsearch_L4_BC_noise'
    path = r'C:\Users\Simo\Laskenta\SimuOut'
    metadata_filename = 'MeanFR__20201203_2029581.csv'
    metadata_fullpath = os.path.join(path,metadata_filename)
    
    df = pd.read_csv(metadata_fullpath, index_col=0)
    index_column = ['Dimension-1 Value', 'Dimension-2 Value']
    df_long = SV.unpivot_dataframe(df, index_column=index_column, kw_sub_to_unpivot='MeanFR')
    print(df_long)

    g = sns.FacetGrid(df_long, col="groupName", col_wrap=2, height=2)    
    g.map(sns.pointplot, "Dimension-1 Value", "MeanFR")

    g2 = sns.FacetGrid(df_long, col="groupName", col_wrap=2, height=2)    
    g2.map(sns.pointplot, "Dimension-2 Value", "MeanFR")
    groups=['MeanFR_NG0_relay_spikes', 'MeanFR_NG1_L4_SS_L4',
       'MeanFR_NG2_L4_BC_L4', 'MeanFR_NG3_L4_PC1_L4toL1']
    group_0 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[0])
    group_1 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[1])
    group_2 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[2])
    group_3 = df.pivot("Dimension-1 Value", "Dimension-2 Value", groups[3])

    plt.figure()
    ax = sns.heatmap(group_0)
    plt.title(groups[0])

    plt.figure()
    ax = sns.heatmap(group_1)
    plt.title(groups[1])

    plt.figure()
    ax = sns.heatmap(group_2)
    plt.title(groups[2])
    
    plt.figure()
    ax = sns.heatmap(group_3)
    plt.title(groups[3])

    plt.show()