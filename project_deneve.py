# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.io as sio

# Computational neuroscience
import brian2.units as b2u
from cxsystem2.core.tools import write_to_file

# Builtin
import os

# Current repo
from system_viz import SystemViz

# Develop
import pdb

'''
Module on project-specific data and analysis.

Prepares data for system_analysis. These data are fed to system_analysis init, where they are processes 
and visualized.

PSEUDOCODE:
get_connections
path in
list Fig4 mat variables

connectivities (dimensions):
FE (3,300) = input group to E = IN2E
CsEE (29, 300, 300) = E2E (the first 29 is development at log intervals of the 300:300 conn matrix)
CsEI (29, 75, 300) = E2I
CsIE (29, 300, 75) = I2E
CsII (29, 75, 75) =I2I

Optimal decoders for each instance of the registered connectivities.
DecsE (29, 3, 300) = E2D
DecI  (29, 3, 75) = I2D

read Fig4 mat files. 

Learn the CxSystem conn data type 
TÄHÄN JÄIT: AJA DENEVE SIMULAATIO OIKEILLA SOLUMÄÄRILLÄ JOTTA SAAT CONN FILEN CONN NIMET JA SOLUJEN PAIKAT

turn conn.mat to conn.gz


path out
save

Developed by Simo Vanni 2021
'''

class Project(SystemViz):

    def __init__(self, path='./', mat_filename=None, connection_filename_out=None, 
                connection_skeleton_filename_in=None):

        self.path = path
        self.mat_filename = mat_filename
        self.connection_filename_out = connection_filename_out
        self.connection_skeleton_filename_in = connection_skeleton_filename_in

    # def get_mat_data_dict(self):

    #     mat_file_name=self.mat_filename
    #     mat_fullpath_filename = self._parsePath(mat_file_name, data_type='.mat')
    #     mat_data_dict = sio.loadmat(mat_fullpath_filename)
    #     return mat_data_dict

    # def get_mat_data_dict(self):

        
    #     return mat_data_dict

    # def get_skeleton_gz_data_dict(self):


    #     return connection_skeleton_dict

    def _show_histogram(self, data, skipzeros=False, bins=10):

        if 'scipy.sparse.csr.csr_matrix' in str(type(data)):
            data = data.toarray()
        
        if 'numpy.ndarray' in str(type(data)):
            data = data.flatten()
        if skipzeros == True:
            data = data[data != 0]
        # pdb.set_trace()
        plt.hist(data, bins=bins);plt.show()

    def replace_connections(self):

        mat_data_dict = self.getData(self.mat_filename)
        connection_skeleton_dict = self.getData(self.connection_skeleton_filename_in)

        mat_keys= ['FsE', 'CsEE', 'CsEI', 'CsIE', 'CsII', 'DecsE', 'DecsI']
        mat_data_dict_keys_str = str(mat_data_dict.keys())

        assert all([x in mat_data_dict_keys_str for x in mat_keys]), 'Some mat keys not found, aborting...'

        match_connection_names = {  'relay_video__to__L4_SS_L4_soma' : 'FsE',
                                    'L4_SS_L4__to__L4_SS_L4_soma' : 'CsEE',
                                    'L4_SS_L4__to__L4_BC_L4_soma' : 'CsEI',
                                    'L4_BC_L4__to__L4_SS_L4_soma' : 'CsIE',
                                    'L4_BC_L4__to__L4_BC_L4_soma' : 'CsII',
                                    'L4_SS_L4__to__L4_SS2_L4_soma' : 'DecsE',
                                    'L4_BC_L4__to__L4_SS2_L4_soma' : 'DecsI'
                                    }

        # which phase of learned connections to select. 29 = after teaching
        mat_teach_idx = 28 
        connection_final_dict = connection_skeleton_dict

        for this_connection in match_connection_names.keys():
            # Get cxsystem connection strengths (i.e. Physiology parameters J, J_I, k*J or k_I*J_I 
            # multiplied by n synapses/connection)
            data_cx = connection_skeleton_dict[this_connection]['data']
            # Get mat_data connection strengths
            data_mat = mat_data_dict[match_connection_names[this_connection]][mat_teach_idx,:,:]
            if data_mat.shape != data_cx.shape:
                pdb.set_trace()
            assert data_mat.shape == data_cx.shape, 'Connection shape mismatch, aborting...'

            # Scale mat_data to min and max values of cxsystem connection strengths (excluding zeros)
            data_out = self.scale_values(data_mat, target_data=data_cx, skip_under_zeros_in_scaling=True)
            # viz by request
            # self._show_histogram(data_cx, skipzeros=True)
            # self._show_histogram(data_mat, skipzeros=False)
            # self._show_histogram(data_out, skipzeros=False)
            # return scaled values
            connection_final_dict[this_connection] = data_out

        pdb.set_trace()

        pass


if __name__=='__main__':

    path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve'
    experiment_folder = 'Replica_test'
    mat_filename = 'Fig4_workspace.mat'
    connection_filename_out = 'connections_Fig4.gz'
    # connection_skeleton_filename_in = 'Replica_test_connections_20210118_1141594.gz'
    connection_skeleton_filename_in = 'Replica_test_connections_20210119_1436570.gz'

    P = Project(
        path=path, 
        mat_filename = mat_filename, 
        connection_filename_out = os.path.join(experiment_folder, connection_filename_out),
        connection_skeleton_filename_in = os.path.join(experiment_folder, connection_skeleton_filename_in))

    # mat_data_dict = P.get_mat_data_dict()
    # connection_skeleton_dict = P.get_skeleton_gz_data_dict()
    P.replace_connections()


    # print(connection_skeleton_dict['positions_all']['w_coord'].keys())
    # print([n for n in dir(mat_data_dict) if not n.startswith('_') ])


    