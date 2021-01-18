# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.io as sio

# Computational neuroscience
import brian2.units as b2u

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

    def get_mat_data_dict(self):

        mat_data_dict = self.getData(self.mat_filename)
        
        return mat_data_dict

    def get_skeleton_gz_data_dict(self):

        connection_skeleton_dict = self.getData(self.connection_skeleton_filename_in)

        return connection_skeleton_dict

if __name__=='__main__':

    path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve'
    experiment_folder = 'Replica_test'
    mat_filename = 'Fig4_workspace.mat'
    connection_filename_out = 'connections_Fig4.gz'
    connection_skeleton_filename_in = r'\Replica_test\Replica_test_connections_20210118_0822512.gz'

    P = Project(path=path, 
                mat_filename=mat_filename, 
                connection_filename_out=connection_filename_out,
                connection_skeleton_filename_in=connection_skeleton_filename_in)

    mat_data_dict = P.get_mat_data_dict()
    connection_skeleton_dict = P.get_skeleton_gz_data_dict()
    print(connection_skeleton_dict.keys())
    # print([n for n in dir(mat_data_dict) if not n.startswith('_') ])


    