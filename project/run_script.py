# Run in correct python environment as python run_script()

# Analysis
from sqlite3.dbapi2 import DataError
from numpy import NaN
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000) # to see it all :)

# This computer git repos
from cxsystem2.core.tools import change_parameter_value_in_file, read_config_file, parameter_finder 

# Builtins
from pathlib import Path

import subprocess
import os
import sys
import time
from multiprocessing import Pool
import logging
import pdb


try:
    from project_deneve import ProjectManager
except:
    raise ModuleNotFoundError('No project_deneve module found, add path to SystemTools to your $PYTHONPATH environment variable')
    
### CHANGE THESE ###
if sys.platform == 'linux':
    base_path = '/opt3/Laskenta/Models/Deneve'
elif sys.platform == 'win32':
    base_path = r'C:\Users\Simo\Laskenta\Models\Deneve'

learned_by = 'Single' # Single or Multi
# path = Path.joinpath(base_path,f'{learned_by}_narrow_iterations/dummies') #  dummy
# path = Path.joinpath(base_path,f'{learned_by}_narrow_iterations') 
path = Path.joinpath(base_path,f'{learned_by}_narrow_iterations_control_ALL') 

# parameters = ['C', 'gL', 'delay', 'EL', 'VT', 'tau_e', 'tau_i']
# parameters = ['C', 'gL', 'EL', 'VT', 'delay']
parameters = ['gL']

## Note: For narrow search csv creation one midpoint at a time
# midpoints = ['Comrad', 'Bacon', 'HiFi']
midpoints = ['Bacon']

# For IxO analyzes only 'NormError', 'Coherence', 'TransferEntropy', 'GrCaus' accepted
# For TEDrift analyzes only 'TEDrift' accepted. 14.12.2021 Transfer entropy with time shifts implemented
# analyzes = ['NormError', 'MeanFR', 'Coherence', 'TransferEntropy', 'Classify','GrCaus']
# analyzes = ['MeanFR'] # Must be analyzed with the 'matched_IO' parallel type
analyzes = ['Coherence', 'TransferEntropy', 'GrCaus', 'NormError'] # Best analyzed with the 'full_IxO' parallel type
# analyzes = ['TEDrift']


t_idx_start=2000
t_idx_end=-2000 # -2000 means 2000 samples from the end will be rejected

create_csvs = 1
run_simulation = 0
run_analysis = 0
# Three parallel analyzes exist (None for non-parallel): 
# 'full_IxO' : create .gz with all inputs vs all outputs, then csv:s. 
# 'matched_IO' : create csv:s from matched input to output pairs. Only GcGaus F-stat classification. Includes some not-so-interesting analyzes missing from full_IXO
# 'TEDrift' : create .gz with input x output x n_parameter_variations x n_timeshifts, then csv. 14.12.2021 applied for transfer entropy latency shifts 
parallel_analysis_type = 'full_IxO' 
create_log_file = True

# Number of input files for simulation. For single input keep n_iters = 1 and  iter_start_idx = 0
n_iters = 10
iter_start_idx = 0
iter_idx_list = [] #[0, 5, 9] # optional, [] for other options. Use this for processing only a subset of files.

anat_update_dict = {
    'workspace_path' : f'{path}',
    # 'import_connections_from': f'../in/connections_{learned_by}Spike_ci.gz',
    # 'import_connections_from': f'../in/connections_{learned_by}Spike_ci_permuted_EI.gz',
    'import_connections_from': f'../in/connections_{learned_by}Spike_ci_permuted_ALL.gz',
    'run_in_cluster' : 0, # SIC(!)
    'cluster_job_file_path' : '../csc_puhti_sv.job'}

configuration = 'narrow' # config
manual_narrow_csv = True # If false, tries to wrap values arounf the wide data peaks. Not very good because max maybe at edge.

# parameter : [variable, key, value, row_by] row_by means df row number by variable or key
noise_timestamp = '210916'# 210406
input_fname_prefix = f'noise_{noise_timestamp}_{learned_by}Spike'
# input_fname_ci_suffix = f'_ci.mat' # f'_permuted_ci.mat'
input_fname_ci_suffix = f'_permuted_ci.mat' # f'_permuted_ci.mat'
## Wide
phys_update_dict = {
    'current_injection' : [['base_ci_path', NaN, "r'/opt3/Laskenta/Models/Deneve/in'",'', 'variable'], ['ci_filename', NaN, f"'{input_fname_prefix}{input_fname_ci_suffix}'",'', 'variable']],
    'C' : [['L4_CI_BC', 'C', '{1|302|30}',' * pF', 'key'], ['L4_CI_SS', 'C', '{1|302|30}',' * pF', 'key']],
    'gL' : [['L4_CI_BC', 'gL', '{1|32|3}',' * nS', 'key'], ['L4_CI_SS', 'gL', '{1|32|3}',' * nS', 'key']],
    'VT' : [['L4_CI_BC', 'VT', '{-100|0|10}',' * mV', 'key'], ['L4_CI_SS', 'VT', '{-100|0|10}',' * mV', 'key']],
    'EL' : [['L4_CI_BC', 'EL', '{-150|0|15}',' * mV', 'key'], ['L4_CI_SS', 'EL', '{-150|0|15}',' * mV', 'key']],
    'RP' : [['L4_CI_BC', 'V_res', '{-100|0|10}',' * mV', 'key'], ['L4_CI_SS', 'RP', '{-100|0|10}',' * mV', 'key']],
    'delay' : [['delay', 'delay_CI_CI', '{0.1|51|0.5}',' * ms', 'key']],
    'tau_e' : [['tau_e_GLOBAL', NaN, '{0.2|15.3|.2}',' * ms', 'variable']],
    'tau_i' : [['tau_i_GLOBAL', NaN, '{0.2|15.3|.2}',' * ms', 'variable']]}
if configuration == 'narrow' and create_csvs == 1 and manual_narrow_csv == False:
    ## Narrow
    assert len(midpoints) == 1, 'Multiple midpoints narrow csv creation not implemented yet'
    path_to_search_limits = Path.joinpath(base_path,f'{learned_by}_wide',f'search_limits_{midpoints[0]}.csv')
    search_limits_df = pd.read_csv(path_to_search_limits)
    for parameter in parameters:
        if len(phys_update_dict[parameter]) == 1:
            axis_dims = ['x']
        elif len(phys_update_dict[parameter]) == 2: 
            axis_dims = ['y', 'x']
        for ax_idx, ax in enumerate(axis_dims):
            exec(f"search_{ax}_{parameter} = search_limits_df.loc[search_limits_df['parameter']=='{parameter}','search_{ax}_string'].values[0]")
            exec(f"phys_update_dict['{parameter}'][{ax_idx}][2] = search_{ax}_{parameter}")

## Put here manual narrow params
if configuration == 'narrow' and create_csvs == 1 and manual_narrow_csv == True:
    phys_update_dict = {
        'current_injection' : [['base_ci_path', NaN, "r'/opt3/Laskenta/Models/Deneve/in'",'', 'variable'], ['ci_filename', NaN, f"'{input_fname_prefix}{input_fname_ci_suffix}'",'', 'variable']],
        'C' : [['L4_CI_BC', 'C', '{ 30.0 | 270.0 | 10.0 }',' * pF', 'key'], ['L4_CI_SS', 'C', '{ 30.0 | 130.0 | 10.0 }',' * pF', 'key']],
        'gL' : [['L4_CI_BC', 'gL', '{ 1.0 | 28.0 | 1.0 }',' * nS', 'key'], ['L4_CI_SS', 'gL', '{ 0.5 | 15.0 | 1.0 }',' * nS', 'key']],
        'VT' : [['L4_CI_BC', 'VT', '{ -65.0 | -15.0 | 3.0 }',' * mV', 'key'], ['L4_CI_SS', 'VT', '{ -67.0 | -35.0 | 3.0 }',' * mV', 'key']],
        'EL' : [['L4_CI_BC', 'EL', '{ -85.0 | -35.0 | 5.0 }',' * mV', 'key'], ['L4_CI_SS', 'EL', '{ -85.0 | -20.0 | 5.0 }',' * mV', 'key']],
        'delay' : [['delay', 'delay_CI_CI', '{ 0.5 | 25 | 0.25 }', ' * ms', 'key']]} 

### END OF CHANGE THESE ###

### CHECK THESE ###
input_folder = '../in'
# input_folder = '../../in' # dummy
workspace_deneve_filename = f'workspace_deneve_{learned_by}Spike.mat'
conn_skeleton_file_in = 'Replica_skeleton_connections_20210211_1453238_L4_SS.gz'
conn_file_out = f'connections_{learned_by}Spike_ci.gz'
# input_filename = f'noise_210406_{learned_by}Spike.mat' #'input_quadratic_three_units_2s_MultiSpike.mat'# 'noise_210406_MultiSpike.mat' # 
input_filename = f'noise_210916_{learned_by}Spike_0.mat' #'input_quadratic_three_units_2s_MultiSpike.mat'# 'noise_210406_MultiSpike.mat' # 
NG_output = 'NG3_L4_SS_L4' 

PM = ProjectManager(path=path, input_folder=input_folder, output_folder=None, 
            workspace_deneve_filename=workspace_deneve_filename, conn_skeleton_file_in=conn_skeleton_file_in, 
            conn_file_out=conn_file_out, input_filename=input_filename, NG_name=NG_output)
# PM = Project(path=path, input_folder='../in', input_filename=input_filename, NG_name='NG3_L4_SS_L4')
PM.t_idx_start=t_idx_start; PM.t_idx_end=t_idx_end

TE_args = {
    'max_time_lag_seconds': 0.1,
    'downsampling_factor': 40,
    'n_states': 4,
    'embedding_vector': 1,
    'te_shift_start_time': 0.004, # only for TEDrift analysis  
    'te_shift_end_time': 0.08}   # only for TEDrift analysis
GrCaus_args = {
    'max_time_lag_seconds': 0.1,
    'downsampling_factor': 40,
    'save_gc_fit_dg_and_QA': False,
    'show_gc_fit_diagnostics_figure': False}  
NormError_args = {
    'decoding_method':'least_squares'} 
kw_ana_args = {
'TE_args': TE_args,
'GrCaus_args': GrCaus_args,
'NormError_args': NormError_args}

# time_ids ={'Comrad':'210410', 'Bacon':'210415', 'HiFi':'210429'} 
time_ids ={'Comrad':'211118', 'Bacon':'211222', 'HiFi':'211118'} # After 211118 the Vm is monitored only in output, resulting in <1% of file size
### END OF CHECK THESE ###


os.chdir(path)


if create_log_file:
    timestamp = time.strftime('%y%m%d_%H%M%S', time.localtime()) # Simple version
    log_full_filename = Path.joinpath(path, f'Script_logfile_{timestamp}.log')
    logging.basicConfig(filename=log_full_filename, encoding='utf-8', level=logging.DEBUG, format='%(levelname)s:%(message)s')

def analyze_TEDrift(PM, analyzes_list):
    PM.analyze_TE_drift(metadata_filename=None, analyzes_list=analyzes_list, **kw_ana_args)

def analyze_IXO(PM, analyzes_list):
    PM.analyze_IxO_array(metadata_filename=None, analyzes_list=analyzes_list, **kw_ana_args)

def analyze(PM, current_analysis):

    if current_analysis == 'NormError':
        PM.analyze_arrayrun(metadata_filename=None, analysis=f'{current_analysis}', **NormError_args)
    elif current_analysis in ['GrCaus', 'Classify']:
        PM.analyze_arrayrun(metadata_filename=None, analysis=f'{current_analysis}', **GrCaus_args)
    elif current_analysis in ['TransferEntropy']:
        PM.analyze_arrayrun(metadata_filename=None, analysis=f'{current_analysis}', **TE_args)
    else:
        PM.analyze_arrayrun(metadata_filename=None, analysis=f'{current_analysis}')

def phys_param_updater(physio_config_df, param_list):

    # Find row index for correct Variable
    index = physio_config_df.index
    condition = physio_config_df["Variable"] == param_list[0]
    variable_index = index[condition].values
    assert len(variable_index) == 1, "Zero or nonunique variable name found, aborting..."

    if param_list[4] == 'variable':
        physio_config_df.loc[variable_index[0],'Value'] = param_list[2] + param_list[3]
    elif param_list[4] == 'key':
        condition_keys = physio_config_df["Key"] == param_list[1]
        key_indices = index[condition_keys].values
        # Find first correct Key after correct Variable. This is dangerous, because it does not check for missing Key
        key_index = key_indices[key_indices >= variable_index][0] 
        physio_config_df.loc[key_index,'Value'] = param_list[2] + param_list[3]
    else:
        raise NotImplementedError('Unknown row_by value, should be "key" or "variable"')
    return physio_config_df

def parallel_analysis(PM, this_idx, idx_iterator, midpoint, parameter, input_fname_prefix, analyzes, parallel_analysis_type):

    if this_idx == 0 and len(idx_iterator) == 1 and not parallel_analysis_type:
        PM.output_folder=f'{midpoint}_{parameter}'
    else: # input filename must be the correct noise input. Alternatively it should be extracted from physio
        PM.output_folder=f'{midpoint}_{parameter}_{this_idx}'
        # PM.output_folder=f'{midpoint}_{parameter}_dummy{this_idx}'
        PM.input_filename = f'{input_fname_prefix}_{str(this_idx)}.mat'
    
    if parallel_analysis_type == 'matched_IO':
        for this_analysis in analyzes:
            try:
                analyze(PM, this_analysis)
            except:
                if create_log_file:
                    logging.error(f'\nParallel matched_IO analysis failed at midpoint {midpoint} parameter {parameter} iteration {this_idx} analysis {this_analysis}')
    elif parallel_analysis_type == 'full_IxO':
        try:
            analyze_IXO(PM, analyzes)
        except:
            if create_log_file:
                logging.error(f'\nParallel IxO analysis failed at midpoint {midpoint} parameter {parameter} iteration {this_idx}')
    elif parallel_analysis_type == 'TEDrift':
        try:
            analyze_TEDrift(PM, analyzes)
        except:
            if create_log_file:
                logging.error(f'\nParallel TEDrift analysis failed at midpoint {midpoint} parameter {parameter} iteration {this_idx}')


if iter_idx_list:
    idx_iterator = iter_idx_list
else:
    idx_iterator = np.arange(iter_start_idx, iter_start_idx + n_iters).tolist()

# loop run
for midpoint in midpoints:
    for parameter in parameters:

        if create_csvs:
            for this_idx in idx_iterator:
                # Anatomy and system
                anatomy_and_system_config = f'Anatomy_config_Deneve_{time_ids[midpoint]}_{midpoint}_midpoint.csv'
                if this_idx == 0 and len(idx_iterator) == 1:
                    anatomy_and_system_config_search = f'Anatomy_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}.csv'
                    change_parameter_value_in_file(
                        anatomy_and_system_config, 
                        anatomy_and_system_config_search, 
                        'simulation_title', 
                        f'{midpoint}_{parameter}')
                else:
                    anatomy_and_system_config_search = f'Anatomy_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}_{this_idx}.csv'
                    change_parameter_value_in_file(
                        anatomy_and_system_config, 
                        anatomy_and_system_config_search, 
                        'simulation_title', 
                        f'{midpoint}_{parameter}_{this_idx}')

                # change_parameter_value_in_file(filepath, save_path, parameter, new_value):
                for this_key in anat_update_dict.keys():
                    change_parameter_value_in_file(
                        anatomy_and_system_config_search, 
                        anatomy_and_system_config_search, 
                        this_key, 
                        anat_update_dict[this_key])

                # Physiology
                physiology_config = f'Physiology_config_Deneve_{time_ids[midpoint]}_{midpoint}_midpoint.csv'
                physio_config_df = read_config_file(physiology_config, header=True)
                if this_idx == 0 and len(idx_iterator) == 1:
                    physiology_config_search = f'Physiology_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}.csv'
                else:
                    physiology_config_search = f'Physiology_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}_{this_idx}.csv'
                
                for param_list in phys_update_dict['current_injection']:
                    # We update the ci_filename to contain the iteration index if multiple noise files requested
                    if param_list[0] == 'ci_filename' and (this_idx != 0 or len(idx_iterator) != 1):
                        param_list[2] = f"'{input_fname_prefix}_{str(this_idx)}{input_fname_ci_suffix}'"
                        physio_config_df = phys_param_updater(physio_config_df, param_list)
                    else:
                        physio_config_df = phys_param_updater(physio_config_df, param_list)
                for param_list in phys_update_dict[parameter]:
                    physio_config_df = phys_param_updater(physio_config_df, param_list)
                physio_config_df.to_csv(physiology_config_search, index=False, header=True)

        if run_simulation:
            try:
                for this_idx in idx_iterator:
                    if this_idx == 0 and len(idx_iterator) == 1: # NOTE that this does not work if you run the 0th iteration alone
                        funcion_call_str = f'''
                        cxsystem2 \
                        -a Anatomy_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}.csv \
                        -p Physiology_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}.csv
                        '''
                        logging.info(funcion_call_str)
                        subprocess.run([funcion_call_str], shell=True)
                    else:
                        funcion_call_str = f'''
                        cxsystem2 \
                        -a Anatomy_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}_{this_idx}.csv \
                        -p Physiology_{configuration}_Deneve_{time_ids[midpoint]}_{midpoint}_{parameter}_{this_idx}.csv
                        '''
                        logging.info(funcion_call_str)
                        subprocess.run([funcion_call_str], shell=True)
            except:
                if create_log_file:
                    logging.error(f'\nSimulation failed midpoint {midpoint} parameter {parameter}')

        if run_analysis:
            tic = time.time()
            if parallel_analysis_type:
                pass_flag = 0
                _tmp_idx_iterator = idx_iterator
                while pass_flag == 0:

                    # Run multiprocessing
                    ncpus = os.cpu_count() - 1
                    pool = Pool(ncpus)
                    for this_idx in _tmp_idx_iterator:
                        res = pool.apply_async(parallel_analysis, (PM, this_idx, _tmp_idx_iterator, midpoint, parameter, input_fname_prefix, analyzes, parallel_analysis_type))
                    pool.close()
                    pool.join()
                    
                    # # debugging without multiprocessing
                    # for this_idx in _tmp_idx_iterator:
                    #     ## With no multiprocessing you call this: 
                    #     parallel_analysis(PM, this_idx, _tmp_idx_iterator, midpoint, parameter, input_fname_prefix, analyzes, parallel_analysis_type)

                    failed_grcaus_iterations = []
                    # Search failed files
                    for this_idx in _tmp_idx_iterator:
                        PM.output_folder=f'{midpoint}_{parameter}_{this_idx}'
                        # PM.output_folder=f'{midpoint}_{parameter}_dummy{this_idx}'
                        tmp_full_filename = Path.joinpath(PM.output_folder, '_tmp_regression_errors.txt')
                        reg_error_filename = Path.joinpath(PM.output_folder, 'regression_errors.txt')
                        if tmp_full_filename.is_file():
                            failed_grcaus_iterations.append(this_idx)
                            # read tmp file and append it to regression errors txt
                            f1 = open(reg_error_filename, 'a+')
                            f2 = open(tmp_full_filename, 'r')
                            f1.write(f2.read())
                            f1.close(); f2.close()
                            Path.unlink(tmp_full_filename)

                    # Set failed files for re-iteration
                    if failed_grcaus_iterations:
                        if create_log_file:
                            logging.error(f'\nGrCaus failed at midpoint {midpoint} parameter {parameter}, iterations {failed_grcaus_iterations}')
                        _tmp_idx_iterator = failed_grcaus_iterations
                    else:
                        pass_flag = 1
            else:
                for this_idx in idx_iterator:
                    if this_idx == 0 and len(idx_iterator) == 1:
                        PM.output_folder=f'{midpoint}_{parameter}'
                    else: # input filename must be the correct noise input. Alternatively it should be extracted from physio
                        PM.output_folder=f'{midpoint}_{parameter}_{this_idx}'
                        PM.input_filename = f'{input_fname_prefix}_{str(this_idx)}.mat'
                    for this_analysis in analyzes:
                        try:
                            analyze(PM, this_analysis)
                        except:
                            if create_log_file:
                                logging.error(f'\nScript analysis failed at midpoint {midpoint} parameter {parameter} analysis {this_analysis}, iteration {this_idx}')
            toc = time.time()
            if create_log_file:
                logging.info(f'\nAnalysis of midpoint {midpoint} parameter {parameter} took {(toc-tic):.2f} seconds')
