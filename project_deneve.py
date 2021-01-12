# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Computational neuroscience
import brian2.units as b2u

# Builtin
import os


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

read Fig4 mat files. Learn the CxSystem conn data type -- turn conn.mat to conn.gz
path out
save

Developed by Simo Vanni 2021
'''

class SystemProject():

    def __init__(self, path='./'):

        self.path=path



if __name__=='__main__':

    path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve'

    den = SystemProject(path=path)


    