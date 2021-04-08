# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import shapiro, normaltest, ttest_1samp
from scipy.signal import decimate, resample
import seaborn as sns

# Statistics
from statsmodels.tsa.stattools import grangercausalitytests as gc_test
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.diagnostic import kstest_normal, het_breuschpagan
from statsmodels.compat import lzip
# import statsmodels.api as sm
from PyIF import te_compute as te

# Computational neuroscience
import brian2.units as b2u
import elephant as el 
from neo.core import AnalogSignal
import quantities as pq

# Builtin
import os
import pickle
import sys
import math

# Current repo
from system_utilities import SystemUtilities


# Develop
import pdb
import warnings
warnings.filterwarnings("ignore")
import time

'''
Module on analysis of simulated electrophysiology data.

Inherits SystemViz which inherits SystemUtilities

Developed by Simo Vanni 2020-2021
'''

class SystemAnalysis(SystemUtilities):

    map_analysis_names = {'meanfr':'MeanFR', 'eicurrentdiff':'EICurrentDiff', 'grcaus':'GrCaus', 'meanvm':'MeanVm'}
    map_data_types = {'meanfr':'spikes_all', 'eicurrentdiff':'vm_all', 'grcaus': 'vm_all', 'meanvm': 'vm_all'}

    def __init__(self, path='./'):

        self.path=path

    def _get_spikes_by_interval(self, data_by_group, t_start, t_end):

        spikes = data_by_group['t'][np.logical_and(data_by_group['t'] > t_start * b2u.second, data_by_group['t'] < t_end * b2u.second)]

        return spikes

    def  _analyze_meanfr(self, data, NG, t_idx_start, t_idx_end):
        
        data_by_group = data['spikes_all'][NG]

        # Get and mark MeanFR to df
        N_neurons = data_by_group['count'].size

        # spikes by interval needs seconds, thus we need to multiply with dt
        dt = self._get_dt(data)
        n_samples = data['time_vector'].shape[0]
        t_idx_end = self._end2idx(t_idx_end, n_samples)

        spikes = self._get_spikes_by_interval(data_by_group, t_start=t_idx_start * dt, t_end=t_idx_end * dt)
        MeanFR = spikes.size / (N_neurons * (t_idx_end - t_idx_start) * dt)

        return MeanFR

    def  _analyze_meanvm(self, data, NG, t_idx_start, t_idx_end):

        data_by_group = data['vm_all'][NG]

        N_neurons = data_by_group['vm'].shape[1]

        vm = self._get_vm_by_interval(data_by_group, t_idx_start=t_idx_start, t_idx_end=t_idx_end)

        MeanVm = np.mean(vm)

        return MeanVm

    def _get_currents_by_interval(self, data, NG, t_idx_start=0, t_idx_end=None):

        ge = data['ge_soma_all'][NG]['ge_soma']
        gi = data['gi_soma_all'][NG]['gi_soma']
        vm = data['vm_all'][NG]['vm']

        # Get necessary variables
        # Calculate excitatory, inhibitory (and leak) currents
        gl = data['Neuron_Groups_Parameters'][NG]['namespace']['gL']
        El = data['Neuron_Groups_Parameters'][NG]['namespace']['EL']         
        I_leak = gl * (El - vm)

        # If no driving force in neuron vm model synapses. This is currently denoted in neuron group name by _CI_ prefix
        if '_CI_' in NG : 
            I_e =  ge * b2u.mV
            I_i =  gi * b2u.mV
        else:
            Ee = data['Neuron_Groups_Parameters'][NG]['namespace']['Ee']
            Ei = data['Neuron_Groups_Parameters'][NG]['namespace']['Ei']         
            I_e =  ge * (Ee - vm)
            I_i =  gi * (Ei - vm)

        return I_e[t_idx_start:t_idx_end,:], I_i[t_idx_start:t_idx_end,:], I_leak[t_idx_start:t_idx_end,:]

    def  _analyze_eicurrentdiff(self, data, NG, t_idx_start=0, t_idx_end=None):

        I_e, I_i, I_leak = self._get_currents_by_interval(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)

        N_neurons = I_e.shape[1]

        # Calculate difference unit by unit, I_e all positive, I_i all negative, thus the difference is +
        EIdifference = I_e[t_idx_start:t_idx_end] + I_i[t_idx_start:t_idx_end]

        MeanEIdifference = np.sum(np.abs(EIdifference)) / N_neurons
 
        return MeanEIdifference
                
    def _get_vm_by_interval(self, data, NG=None, t_idx_start=0, t_idx_end=None):
        
        if NG is None:
            vm = data['vm'] # data_by_group already
        else:
            vm = data['vm_all'][NG]['vm']

        return vm[t_idx_start:t_idx_end,:]

    def _test_time_lag(self, data, max_lag, dt, downsampling_factor):
        
        #instantiate the VAR function on time series data
        model = VAR(data)

        max_lag_samples = int(max_lag / (dt * downsampling_factor))

        for i in ['aic', 'bic', 'hqic']:
            #maxlags takes the number of lags we want to test
            #ic takes the information criterion method based on which order would be suggested
            try:
                results = model.fit(maxlags=max_lag_samples, ic=i, trend='c')
            except ValueError:
                print(f'Error: maxlags {max_lag_samples} is too large for the number of observations {data.shape[0]}')
                sys.exit()
            order = results.k_ar

            print(f"The suggested VAR order from {i} is {order}")

            #To test absence of significant residual autocorrelations one can use the test_whiteness method of VARResults
            test_corr = results.test_whiteness(nlags=max_lag_samples + 1, signif=0.05, adjusted=False)

            ##Print the p-value
            ##There is no serial autocorrelation in residuals if p-value is more than 0.05
            p_value = test_corr.pvalue
            if p_value > 0.05:
                print(f'WARNING: Serial autocorrelation test failed, p = {p_value}')
            else:
                print('Serial autocorrelation test passed')
            
            # print(results.summary())

    def _return_best_order(self, data, max_lag, ic, dt, downsampling_factor):

        max_lag_samples = int(max_lag / (dt * downsampling_factor))
        model = VAR(data)
        results = model.fit(maxlags=max_lag_samples, ic=ic, trend='c')
        best_time_lag_samples = results.k_ar

        return best_time_lag_samples

    def transfer_entropy(self, data, max_lag):
        '''
        FOR UNKNOWN REASON THIS DOES NOT SEEM TO GIVE ANYTHING USEFUL
        IT SEES ALL INPUT OUTPUT PAIRS AS EQUAL IN TE SENSE. FOR SATURATED
        SPIKE TRAINS IN GROUP 1 IT GIVES NEGATIVE VALUES 
        Measures Information transmitted from Y to X

        data: N x 2 matrix, where
        N is the number of samples
        data[:,0] is target (X)
        data[:,1] is source (Y)
        k: controls the number of neighbors used in KD-tree queries. Keep 1 (best resolution)
        embedding: controls how many lagged periods are used to estimate transfer entropy
        GPU: a boolean argument that indicates if CUDA compatible GPUs should be used to estimate transfer entropy instead of your computer's CPUs.
        safetyCheck: a boolean argument can be used to check for duplicates rows in your dataset. Keep True
        '''

        transfer_entropy = te.te_compute(data[:,0], data[:,1], k=1, embedding=max_lag, safetyCheck=True, GPU=False)

        return transfer_entropy 

    def vm_entropy(self, data, base=None, bins=None):
        """ Computes entropy of data distribution. """

        #de-mean data
        data_demeaned = data - data.mean(axis=0)
        #flatten data
        _data = data_demeaned.flatten()
        n_labels = len(_data)

        if n_labels <= 1:
            return 0

        # Assuming Vm dynamic range about 30 mV. Assuming a 10-step resolution 
        # Note, demeaned.   
        bins = np.linspace(-15,15,11) 

        if bins is not None:
            counts,foo = np.histogram(_data,bins=bins)
        else:
            value,counts = np.unique(_data, return_counts=True)

        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.

        # Compute entropy
        base = math.e if base is None else base

        for i in probs:
            # ent change is 0 if P(i) is 0, ie empty bin
            if i == 0.0:
                pass
            else:
                ent -= i * math.log(i, base)

        if 0:
            plt.hist(_data,bins=bins)
            plt.title(f'vm_entropy = {ent}')
            plt.show()

        return ent

    def downsample(self, data, downsampling_factor=1, axis=0):
        # Note, using finite impulse response filter type
        # downsampled_data = decimate(data, downsampling_factor, axis=axis, ftype='fir',)
        N_time_points = data.shape[0]
        num = N_time_points // downsampling_factor
        downsampled_data =  resample(data, num)
        # pdb.set_trace()
        return downsampled_data

    def _get_passing(self, value, threshold, passing_goes='over'):
        assert passing_goes in ['under', 'over', 'both'], \
            'Unkown option for passing_goes parameter, valid options are "over" and "under"'

        if np.isnan(value):
                passing = 'FAIL'        
        elif passing_goes=='over':
            if value <= threshold:
                passing = 'FAIL'
            elif value > threshold:
                passing = 'PASS'
        elif passing_goes=='under':
            if value > threshold:
                passing = 'FAIL'
            elif value <= threshold:
                passing = 'PASS'
        elif passing_goes=='both':
            if threshold[0] <= value <= threshold[1]:
                passing = 'PASS'
            else:
                passing = 'FAIL'

        return passing

    def _gc_model_diagnostics(self, gc_stat_dg_data, pre_idx, post_idx, show_figure=False, save_gc_fit_diagnostics=False):

        full_model = 1
        partial_model = 0
        current_model = full_model

        _results = gc_stat_dg_data[current_model]
        _residuals = _results.resid
        _exog = gc_stat_dg_data[current_model].model.exog
        _endog = gc_stat_dg_data[current_model].model.endog

        if show_figure is True:
            # plot preparations
            fig = plt.figure()
            plt.rc("figure", figsize=(18,12))
            plt.rc("font", size=10)

        # Heteroscedasticity
        # Homoscedasticity is assumed as 0 hypothesis.
        p_th = 0.001
        stat_het, p_het, foo1, foo2 = het_breuschpagan(_endog, _exog)
        het_passing = self._get_passing(p_het, p_th, passing_goes='over')
        stat_het = self.round_to_n_significant(stat_het)
        p_het = self.round_to_n_significant(p_het)
        if show_figure is True:
            ax12 = plt.subplot(3, 2, (1,2))
            # plt.plot(_exog[:,:-2],_residuals,'.b') # We skip the last regressor from exog, which is constant 1
            plt.plot(_exog[:,:-2],_endog,'.b') # We skip the last regressor from exog, which is constant 1
            ax12.set_title(f'''Heteroscedasticity {het_passing}\npre: {pre_idx}; post: {post_idx},\nstat: {stat_het}; p: {p_het}''')

        # Cook's distance, Influence
        # cook_cutoff = 4 / _results.nobs
        cook_cutoff = 1
        Influence = OLSInfluence(_results)
        cooks_distance = Influence.cooks_distance[0]
        cook_passing = self._get_passing(np.max(cooks_distance), cook_cutoff, passing_goes='under')
        if show_figure is True:
            ax3 = plt.subplot(3, 2, 3)
            Influence.plot_influence(ax=ax3, alpha=0.001)
            ax3.set_title(f'Influence plot (H leverage vs Studentized residuals)')
            ax3.set_xlabel('', fontsize=10)
            ax3.set_ylabel('Studentized residuals', fontsize=10)

            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(cooks_distance,'.b')
            ax4.hlines(cook_cutoff,0,len(cooks_distance), colors='red', linestyles='dashed')
            ax4.set_title(f'''Cook's distance {cook_passing}\nmax value: {np.max(cooks_distance)}''')

        # # Normality
        # # If the pvalue is lower than threshold, we can reject the 
        # # Null hypothesis that the sample comes from a normal distribution.
        p_th = 0.001
        stat_norm, p_norm = kstest_normal(_residuals)
        stat_norm = self.round_to_n_significant(stat_norm)
        p_norm = self.round_to_n_significant(p_norm)
        normality_passing = self._get_passing(p_norm, p_th, passing_goes='over')

        # print(f'Normality of error distribution: p_norm = {p_norm} -- {normality_passing}')
        if show_figure is True:
            ax5 = plt.subplot(3, 2, 5)
            sns.distplot(_residuals, ax=ax5)
            ax5.set_title(f'Distribution of residuals')
            ax6 = plt.subplot(3, 2, 6)
            qqplot(_residuals, line='s', ax=ax6)
            ax6.set_title(f'Normality of residuals {normality_passing}')

        # Serial autocorrelation
        limits = [1.5, 2.5]
        stat_acorr = durbin_watson(_residuals)
        acorr_passing = self._get_passing(stat_acorr, limits, passing_goes='both')
        # print(f'Autocorrelation of error distribution: stat = {str(stat_acorr)} -- {acorr_passing}')

        # Multicollinearity
        # Variance inflation factor
        VIF = np.zeros_like(_results.model.exog[1])
        for this_idx in np.arange(len(_results.model.exog[1])):
            VIF[this_idx] = variance_inflation_factor(_results.model.exog,this_idx)
        max_VIF = np.max(VIF)
        VIF_th = 1000
        vif_passing = self._get_passing(max_VIF, VIF_th, passing_goes='under')
 
        plt.show()

        # Results summary to txt file
        if save_gc_fit_diagnostics is True:
            filename_full_path = self.gc_dg_filename
            results_summary = _results.summary()
            with open(filename_full_path, "a") as text_file:
                text_file.write(f'\n{self.most_recent_loaded_file}\npre {pre_idx} -- post {post_idx}\n')
                text_file.write('\n')
                text_file.write(f'het {het_passing}, cook {cook_passing}, norm {normality_passing}, acorr {acorr_passing}, vif {vif_passing}')
                text_file.write('\n\n')
                text_file.write(str(results_summary))
                text_file.write('\n')

        return het_passing, cook_passing, normality_passing, acorr_passing, vif_passing

    def _end2idx(self, t_idx_end, n_samples):

        if t_idx_end is None:
            t_idx_end = n_samples
        elif t_idx_end < 0:
            t_idx_end = n_samples + t_idx_end + 1
        return t_idx_end

    def _analyze_grcaus(self, data, source_signal, dt, NG, 
                        t_idx_start=0, t_idx_end=None, **kwargs):
        '''
        Get input and output timeseries.
        Run grangercausality for relevant pairs. 
        '''

        max_time_lag_seconds = kwargs['max_time_lag_seconds']
        downsampling_factor = kwargs['downsampling_factor'] 
        test_timelag = kwargs['test_timelag'] 
        do_bonferroni_correction = kwargs['do_bonferroni_correction'] 
        gc_significance_level = kwargs['gc_significance_level'] 
        save_gc_fit_diagnostics = kwargs['save_gc_fit_diagnostics']
        show_figure = kwargs['show_gc_fit_diagnostics_figure']  

        # vm_unit = self._get_vm_by_interval(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)
        vm_unit = self._get_vm_by_interval(data, NG, t_idx_start=0, t_idx_end=-1) # full length for fourier downsampling

        target_signal = vm_unit / vm_unit.get_best_unit() 
        target_signal_entropy = self.vm_entropy(target_signal, base=2)

        assert isinstance(downsampling_factor,int), 'downsampling_factor must be integer, aborting...'
        # Barnett_2017_JNeurosciMeth: sample-period should be kept at 1/10 or less of
        # causal effect delay for detectability sweet spot.
        source_signal_ds = self.downsample(source_signal, downsampling_factor=downsampling_factor, axis=0)
        target_signal_ds = self.downsample(target_signal, downsampling_factor=downsampling_factor, axis=0) 
        # source_signal_ds = source_signal[::downsampling_factor,:]
        # target_signal_ds = target_signal[::downsampling_factor,:]
        # pdb.set_trace()

        # Preprocessing. Preferentially first-order differencing.
        diff_order = 1
        source_signal_pp = np.diff(source_signal_ds, n=diff_order, axis=0)
        target_signal_pp = np.diff(target_signal_ds, n=diff_order, axis=0)

        # Slice source signal to requested time interval
        t_idx_end = self._end2idx(t_idx_end, vm_unit.shape[0])

        
        # Set start and end to nearest allowed integer
        # TÄHÄN JÄIT
        
        # Cut to requested length after preprocessing
        source_signal_pp = source_signal_pp[t_idx_start // downsampling_factor:t_idx_end // downsampling_factor,:]
        target_signal_pp = target_signal_pp[t_idx_start // downsampling_factor:t_idx_end // downsampling_factor,:]

        # time_vector = np.arange(t_idx_start,t_idx_end)
        # plt.plot(time_vector, target_signal[t_idx_start:t_idx_end,0])
        # # plt.plot(time_vector[::downsampling_factor], target_signal_ds[t_idx_start // downsampling_factor:t_idx_end // downsampling_factor,0])
        # plt.plot(time_vector[::downsampling_factor], target_signal_ds[:,0])
        # plt.show()

        pre_idx_array =  np.arange(source_signal_pp.shape[1])
        post_idx_array =  np.arange(target_signal_pp.shape[1])

        # Analyze causal effect delay and exit
        if test_timelag is True:
            for pre_idx in pre_idx_array:
                for post_idx in post_idx_array:
                    _source, _target = source_signal_pp[:,pre_idx], target_signal_pp[:,post_idx]
                    signals = np.vstack([_target, _source]).T
                    self._test_time_lag(signals, max_time_lag_seconds, dt, downsampling_factor)
            # sys.exit()

        gc_matrix_np_F = np.full_like(np.empty((source_signal_pp.shape[1], target_signal_pp.shape[1])),0)
        gc_matrix_np_p = np.full_like(np.empty((source_signal_pp.shape[1], target_signal_pp.shape[1])),np.nan) 
        gc_matrix_np_te = np.full_like(np.empty((source_signal_pp.shape[1], target_signal_pp.shape[1])),np.nan) 
        gc_matrix_np_latency = np.full_like(np.empty((source_signal_pp.shape[1], target_signal_pp.shape[1])),np.nan)
        gc_matrix_np_stationary = np.full_like(np.empty((source_signal_pp.shape[1], target_signal_pp.shape[1])),np.nan) 
        gc_fitQA = 0

        # for each source signal, calculate all target signals.
        for pre_idx in pre_idx_array:
            for post_idx in post_idx_array:
                # print(f'pre {pre_idx} post {post_idx}\r', end="")

                # The model order is the maximum number of lagged observations included in the model
                _source, _target = source_signal_pp[:,pre_idx], target_signal_pp[:,post_idx]
                signals = np.vstack([_target, _source]).T

                # Get best order with Bayesian information criterion
                try:
                    best_time_lag_samples = self._return_best_order(signals, max_time_lag_seconds, 'bic', dt, downsampling_factor)
                except:
                    continue

                pairwise_gc_dict = gc_test(signals, [best_time_lag_samples], verbose=True)

                transfer_entropy_value = self.transfer_entropy(signals, best_time_lag_samples)
                # GC quality control, several measures on error distribution
                gc_stat_dg_data = pairwise_gc_dict[best_time_lag_samples][1]
                het_passing, cook_passing, normality_passing, acorr_passing, vif_passing = \
                    self._gc_model_diagnostics(gc_stat_dg_data, pre_idx, post_idx, 
                    show_figure=show_figure, save_gc_fit_diagnostics=save_gc_fit_diagnostics)
                gc_fitQA += str([het_passing, cook_passing, normality_passing, acorr_passing, vif_passing]).count('PASS')

                # dict_keys(['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'])
                # test statistic, pvalues, degrees of freedom
                gc_test_type = 'ssr_ftest'
                gc_matrix_np_F[pre_idx, post_idx] = pairwise_gc_dict[best_time_lag_samples][0][gc_test_type][0]
                gc_matrix_np_p[pre_idx, post_idx] = pairwise_gc_dict[best_time_lag_samples][0][gc_test_type][1]
                gc_matrix_np_te[pre_idx, post_idx] = transfer_entropy_value
                gc_matrix_np_latency[pre_idx, post_idx] = best_time_lag_samples * dt * downsampling_factor
        
        if do_bonferroni_correction is True:
            corrected_gc_significance_level = gc_significance_level /  gc_matrix_np_p.size
        else:
            corrected_gc_significance_level = gc_significance_level
        
        # Select one-to-one connectivity from input to "correct" output 
        gc_eye_idx = np.eye(source_signal.shape[1], target_signal.shape[1], dtype=bool)

        # Magnitude in base 2 log. If error distribution is gaussian, can be interpreted as bits
        # Multiplied with sample frequency, gives information transfer rate. Lionell Barnett, personal communication
        final_sampling_frequency = 1 / (dt * downsampling_factor)
        gc_matrix_np_InfoRate = np.log2(gc_matrix_np_F) * final_sampling_frequency 
        gc_InfoRate = np.nan_to_num(gc_matrix_np_InfoRate[gc_eye_idx], nan=0.0)
        gc_p = gc_matrix_np_p[gc_eye_idx] 
        gc_latency = gc_matrix_np_latency[gc_eye_idx] 

        # Get representative value
        MeanGrCaus_InfoRate = np.floor(np.nanmean(gc_InfoRate))
        MedianGrCaus_p = np.nanmedian(gc_p)
        PassGrCaus_p = np.nanmedian(gc_p) <=  corrected_gc_significance_level
        MeanGrCaus_latency = np.nanmean(gc_latency)
        MeanGrCaus_fitQA = gc_fitQA / (pre_idx_array.size * post_idx_array.size)

        if 0:
            print(f'BG log2(F): {MeanGrCaus_InfoRate}')
            print(f'BG latency: {MeanGrCaus_latency}')
            print(f'gc_matrix_np_p: \n{gc_matrix_np_p}')  
            # print(f'gc_matrix_np_te: \n{gc_matrix_np_te}')  
            print(f'target_signal_entropy: \n{target_signal_entropy}')
            print(f'significance test:')
            if PassGrCaus_p: print('\033[32mPASS\033[97m')
            else: print('\033[31mFAIL\033[97m')
            print(f'fit quality: {MeanGrCaus_fitQA}')
            print('***End of analysis***\n\n')

        # TODO Calculate CV of gc "grandmother index"

        # Return one value per analysis (mean of best matching units), indicating GrCaus relation
        return MeanGrCaus_InfoRate, MedianGrCaus_p, MeanGrCaus_latency, target_signal_entropy, MeanGrCaus_fitQA

    def get_analyzed_array_as_df(self, data_df, analysisHR=None, t_idx_start=0, t_idx_end=None, **kwargs):
    
        # Get neuron group names
        filename_0 = data_df['Full path'].values[0]
        data = self.getData(filename_0)
        NG_list = [n for n in data[self.map_data_types[analysisHR.lower()]].keys() if 'NG' in n]

        # Add neuron group columns
        if analysisHR.lower() in ['meanfr', 'meanvm', 'eicurrentdiff']:
            for NG in NG_list:
                data_df[f'{analysisHR}_' + NG] = np.nan
        elif analysisHR.lower() in ['grcaus']:
            target_group = self.NG_name            
            data_df[f'{analysisHR}_' + target_group + '_InfoRate'] = np.nan
            data_df[f'{analysisHR}_' + target_group + '_p'] = np.nan
            data_df[f'{analysisHR}_' + target_group + '_latency'] = np.nan
            data_df[f'{analysisHR}_' + target_group + '_target_entropy'] = np.nan
            data_df[f'{analysisHR}_' + target_group + '_fit_quality'] = np.nan
            # Get reference data for granger causality
            analog_input = self.getData( self.input_filename, data_type=None)
            source_signal = analog_input['stimulus'].T # We want time x units
            source_signal_dt = analog_input['frameduration']

        dt = self._get_dt(data)
        
        # Get duration
        if t_idx_end is None:
            t_idx_end = int(data['runtime']  / dt)

        # Loop through datafiles
        for this_index, this_file in zip(data_df.index, data_df['Full path'].values):
            data = self.getData(this_file)
            # Loop through neuron groups 
            if analysisHR.lower() in ['meanfr', 'meanvm', 'eicurrentdiff']:
                for NG in NG_list:
                    # _analyze_meanfr or _analyze_eicurrentdiff, analysis by single group
                    analyzed_results = eval(f'self._analyze_{analysisHR.lower()}(data, NG, t_idx_start=t_idx_start, t_idx_end=t_idx_end)')
                    data_df.loc[this_index,f'{analysisHR}_' + NG] = analyzed_results
            # _analyze__grcaus, analysis between two groups
            elif analysisHR.lower() in ['grcaus']:
                # check how multivariate gc is analyzed; are min, max, mean, median useful?
                # Apply this to _analyze_grangercausality
                MeanGrCaus_InfoRate, MedianGrCaus_p, MeanGrCaus_latency, target_entropy, MeanGrCaus_fitQA = \
                    self._analyze_grcaus(data, source_signal, dt, target_group, t_idx_start=t_idx_start, 
                    t_idx_end=t_idx_end, **kwargs)

                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_InfoRate'] = MeanGrCaus_InfoRate
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_p'] = MedianGrCaus_p
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_latency'] = MeanGrCaus_latency
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_target_entropy'] = target_entropy
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_fit_quality'] = MeanGrCaus_fitQA

        return data_df

    def analyze_arrayrun(self, metadata_filename=None, analysis=None, t_idx_start=0, t_idx_end=None, **kwargs):
        '''
        Create mean firing rate csv table for array run. Needs a metadata file.
        '''
        # Map to standard camelcase
        assert analysis.lower() in self.map_analysis_names.keys(), 'Analysis type not found, aborting...'
        analysisHR = self.map_analysis_names[analysis.lower()]

        # Replace metadata with scalar value for MeanFR or EICurrentDiff
        metadata_fullpath_filename = self._parsePath(metadata_filename, data_type='metadata')
        metadataroot, metadataextension = os.path.splitext(metadata_fullpath_filename)
        filename_out = metadataroot.replace('metadata', analysisHR)
        csv_name_out = filename_out + '.csv'

        if analysisHR.lower() in ['grcaus']:
            save_gc_fit_diagnostics = kwargs['save_gc_fit_diagnostics']
            if save_gc_fit_diagnostics is True:
                diag_filename = 'grcaus_FitDiag.txt'
                self.gc_dg_filename = os.path.join(self.path, diag_filename)
                # Flush existing file empty
                open(self.gc_dg_filename, 'w').close()

        data_df = self.getData(metadata_filename, data_type='metadata')

        analyzed_data_df = self.get_analyzed_array_as_df(
            data_df, analysisHR=analysisHR, t_idx_start=t_idx_start, t_idx_end=t_idx_end, **kwargs)

        # Drop Full path column for concise printing
        analyzed_data_df = analyzed_data_df.drop(['Full path'], axis=1)

        # # Display values
        self.pp_df_full(analyzed_data_df)

        analyzed_data_df.to_csv(csv_name_out, index=True)

    def analyze_plasticity(self, n_iter=1):
        '''
        Run three experiments of plasticity from Clopath_2010_NatNeurosci
        Voltage clamp, STDP and Burst
        '''

        # Voltage clamp

        # Pseudocode
        # Run array experiment. Monitor wght
        # Get results
        # Display


if __name__=='__main__':

    # path = r'C:\Users\Simo\Laskenta\SimuOut\Deneve\Replica_test'

    # analysis = SystemAnalysis(path=path)
    # NG_name = 'NG3_L4_SS2_L4'

    # analysis.plot_readout_on_input(NG_name, normalize=False, filename='Replica_test_results_20210114_1750000.gz')    
    # analysis.show_spikes(filename='Replica_test_results_20210114_1750000.gz')
    
    # plt.show()
    pass