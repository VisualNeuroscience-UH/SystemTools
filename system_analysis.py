# Analysis
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import shapiro, normaltest, ttest_1samp
from scipy.signal import decimate, resample, coherence, csd, correlate, welch
import seaborn as sns

# Statistics
from statsmodels.tsa.stattools import grangercausalitytests as gc_test
from statsmodels.tsa.stattools import adfuller, ccf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.diagnostic import kstest_normal, het_breuschpagan
from statsmodels.compat import lzip
# import statsmodels.api as sm

import pyinform as pin

# Machine learning
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.decomposition import PCA

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

    map_analysis_names = {  'meanfr':'MeanFR', 'eicurrentdiff':'EICurrentDiff', 'grcaus':'GrCaus', 
                            'meanvm':'MeanVm', 'coherence':'Coherence', 'meanerror':'MeanError', 'classify':'Classify'}
    map_data_types = {  'meanfr':'spikes_all', 'eicurrentdiff':'vm_all', 'grcaus': 'vm_all', 
                        'meanvm': 'vm_all', 'coherence': 'vm_all', 'meanerror': 'vm_all', 'classify':'vm_all'}

    def __init__(self, path='./'):

        self.path=path

    def scaler(self, data, scale_type='standard', feature_range=[-1, 1]):
        # From sklearn.preprocessing
        # Data is assumed to be [samples or time, features or regressors]
        if scale_type == 'standard':
            # Standardize data by removing the mean and scaling to unit variance
            data_scaled = scale(data)
        elif scale_type == 'minmax':
            # Transform features by scaling each feature to a given range.
            # If you put in matrix, note that scales each column (feature) independently
            minmaxscaler = MinMaxScaler(feature_range=feature_range)
            minmaxscaler.fit(data)
            data_scaled = minmaxscaler.transform(data)
        return data_scaled

    def _get_spikes_by_interval(self, data_by_group, t_start, t_end):

        spikes = data_by_group['t'][np.logical_and(data_by_group['t'] > t_start * b2u.second, data_by_group['t'] < t_end * b2u.second)]

        return spikes

    def _analyze_meanfr(self, data, NG, t_idx_start, t_idx_end):
        
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

    def _analyze_meanvm(self, data, NG, t_idx_start, t_idx_end):

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

    def _analyze_eicurrentdiff(self, data, NG, t_idx_start=0, t_idx_end=None):

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

        n_samples =vm.shape[0]

        t_idx_end = self._end2idx(t_idx_end, n_samples)

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

    def _return_best_order(self, data, max_lag_seconds, ic, dt, downsampling_factor):

        max_lag_samples = int(max_lag_seconds / (dt * downsampling_factor))
        model = VAR(data)
        results = model.fit(maxlags=max_lag_samples, ic=ic, trend='c')
        best_time_lag_samples = results.k_ar

        return best_time_lag_samples

    def granger_causality(self, target_signal_pp, source_signal_pp, max_lag_seconds, ic, dt, downsampling_factor, signif=0.05, verbose=True):

        max_lag_samples = int(max_lag_seconds / (dt * downsampling_factor))
        def get_column_names(data, prefix):
            try:
                name_list = [f'{prefix}{i}' for i in range(data.shape[1])]
            except IndexError:
                name_list = [f'{prefix}0']
            return name_list
        target_name_list = get_column_names(target_signal_pp, 'target')
        source_name_list = get_column_names(source_signal_pp, 'source')

        # make df for explicit naming
        df = pd.DataFrame(data=np.column_stack([target_signal_pp, source_signal_pp]),columns=target_name_list + source_name_list)
        model = VAR(df)
        results = model.fit(maxlags=max_lag_samples, ic=ic, trend='c')
        best_time_lag_samples = results.k_ar

        gc_results=results.test_causality(target_name_list, causing=source_name_list, kind='f', signif=signif)

        F_value = gc_results.test_statistic
        p_value = gc_results.pvalue

        if verbose is True:
            if p_value <= signif : 
                print(f'\033[32m{gc_results.summary()})\033[97m')
            else: 
                print(f'\033[31m{gc_results.summary()})\033[97m')
        
        return F_value, p_value, best_time_lag_samples

    def pin_transfer_entropy(self, target, source, best_time_lag_samples, base=2):

        '''
        Pyinform implementation of transfer_entropy. It gives the average transfer of entropy per timepoint.
        The local option enables looking at timepointwise transfer
        Start from https://elife-asu.github.io/PyInform/dist.html
        Ref: Moore et al. Front Rob AI, June 2018 | Volume 5 | Article 60
        Ref: J.T. Lizier et al. Information Sciences 208 (2012) 39–54
        '''
        def _data2bins(data, base):
            # Base to bin edges
            bins = np.linspace(np.min(data), np.max(data),num=base + 1, endpoint=True)
            # Get digitized values
            inds = np.digitize(data, bins)
            return inds

        binned_source = _data2bins(source, base)
        binned_target = _data2bins(target, base)
        
        pin_te = pin.transferentropy.transfer_entropy(binned_source, binned_target, best_time_lag_samples, condition=None, local=False)
        return pin_te

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

        return downsampled_data

    def _get_passing(self, value, threshold, passing_goes='over'):
        assert passing_goes in ['under', 'over', 'both'], \
            'Unknown option for passing_goes parameter, valid options are "over" and "under"'

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

    def _correlation_lags(self, in1_len, in2_len, mode='full'):
        # Copied from scipy.signal correlation_lags, mode full or same

        if mode == 'full':
            lags = np.arange(-in2_len + 1, in1_len)        
        elif mode == 'same':
            # the output is the same size as `in1`, centered
            # with respect to the 'full' output.
            # calculate the full output
            lags = np.arange(-in2_len + 1, in1_len)
            # determine the midpoint in the full output
            mid = lags.size // 2
            # determine lag_bound to be used with respect
            # to the midpoint
            lag_bound = in1_len // 2
            # calculate lag ranges for even and odd scenarios
            if in1_len % 2 == 0:
                lags = lags[(mid-lag_bound):(mid+lag_bound)]
            else:
                lags = lags[(mid-lag_bound):(mid+lag_bound)+1]

        return lags

    def _get_error(self, input_signal, output_signal, dt, shift_in_seconds = 0, epoch_in_seconds=1.5):

        shift_in_samples = int(shift_in_seconds / dt)
        epoch_in_samples = int(epoch_in_seconds / dt)

        assert len(input_signal) == len(output_signal), 'Unequal sample lengths, aborting...'
        nsamples = len(input_signal)

        # If shift makes no sense/too long, return -1. Probably noise.
        if  shift_in_samples + epoch_in_samples > nsamples:
            return -1

        output_start_idx = shift_in_samples
        output_end_idx = shift_in_samples + epoch_in_samples
        output_shifted_backwards = output_signal[output_start_idx:output_end_idx]
        input_cut = input_signal[:epoch_in_samples] # input does not shift

        error = self.get_normalized_error_variance(input_cut, output_shifted_backwards)

        # # Deneve error
        # error = np.var(input_cut - output_shifted_backwards) / np.var(input_cut)

        if 0:
            plt.plot(input_cut)
            plt.plot(output_shifted_backwards)
            self._string_on_plot(plt.gca(), variable_name='error', variable_value=error, variable_unit='unitless')
            plt.show()
        return error

    def get_coherence_of_two_signals(self, x, y, samp_freq=1.0, nperseg=64, high_cutoff=100):
        '''
        Coherence, cross spectral density, cross correlation

        From scipy-signal.coherence documentation: Cxy = abs(Pxy)**2/(Pxx*Pyy), 
        where Pxx and Pyy are power spectral density estimates of X and Y, and 
        Pxy is the cross spectral density estimate of X and Y.
        '''
        x_scaled = self.scaler(x)
        y_scaled = self.scaler(y)

        # Coherence
        f, Cxy = coherence(x_scaled, y_scaled, fs=samp_freq, window='hann', nperseg=nperseg)
        
        # Cross spectral density
        f, Pxy = csd(x_scaled, y_scaled, fs=samp_freq, nperseg=nperseg, scaling='density')

        # Cross correlation
        corr = correlate(y_scaled, x_scaled, mode='full')
        lags = self._correlation_lags(len(x_scaled), len(y_scaled))
        corr /= np.max(corr)

        # Power spectrum of input and output signals
        f, Pwelch_spec_x = welch(x_scaled, fs=samp_freq, nperseg=nperseg, scaling='density')
        f, Pwelch_spec_y = welch(y_scaled, fs=samp_freq, nperseg=nperseg, scaling='density')

        if high_cutoff is not None:
            indexes = f < high_cutoff
            f = f[indexes]
            Cxy = Cxy[indexes] 
            Pxy = Pxy[indexes] 
            Pwelch_spec_x = Pwelch_spec_x[indexes] 
            Pwelch_spec_y = Pwelch_spec_y[indexes] 
            
        coherence_sum = np.sum(Cxy)

        return f, Cxy, Pwelch_spec_x, Pwelch_spec_y, Pxy, lags, corr, coherence_sum, x_scaled, y_scaled

    def _analyze_grcaus(self, data, source_signal, dt, target_group, verbose=True, return_only_infomatrix=False, 
                        t_idx_start=0, t_idx_end=None, **kwargs):
        '''
        Granger causality analysis between source and target of correctly classified units
        Assuming that the correctly classified units are on the diagonal indexes [0->0, 1->1 ..., n->n]
        '''

        max_time_lag_seconds = kwargs['max_time_lag_seconds']
        downsampling_factor = kwargs['downsampling_factor'] 
        test_timelag = kwargs['test_timelag'] 
        do_bonferroni_correction = kwargs['do_bonferroni_correction'] 
        gc_significance_level = kwargs['gc_significance_level'] 
        save_gc_fit_diagnostics = kwargs['save_gc_fit_diagnostics']
        show_figure = kwargs['show_gc_fit_diagnostics_figure']  

        vm_unit = self._get_vm_by_interval(data, target_group, t_idx_start=0, t_idx_end=-1) # full length for fourier downsampling
        target_signal = vm_unit / vm_unit.get_best_unit() 
        target_signal_entropy = self.vm_entropy(target_signal, base=2)

        assert isinstance(downsampling_factor,int), 'downsampling_factor must be integer, aborting...'
        # Barnett personal communication: sample-period should be kept reasonable in relation to information transfer.
        source_signal_ds = self.downsample(source_signal, downsampling_factor=downsampling_factor, axis=0)
        target_signal_ds = self.downsample(target_signal, downsampling_factor=downsampling_factor, axis=0) 

        # Preprocessing. Preferentially first-order differencing.
        diff_order = 1
        source_signal_pp = np.diff(source_signal_ds, n=diff_order, axis=0)
        target_signal_pp = np.diff(target_signal_ds, n=diff_order, axis=0)

        # Turn sample idx relative to end, such as -100, to actual end sample idx
        t_idx_end = self._end2idx(t_idx_end, vm_unit.shape[0])
        
        # Set start and end to nearest allowed integer
        # TODO
        
        # Cut to requested length after preprocessing
        source_signal_pp = source_signal_pp[t_idx_start // downsampling_factor:t_idx_end // downsampling_factor,:]
        target_signal_pp = target_signal_pp[t_idx_start // downsampling_factor:t_idx_end // downsampling_factor,:]

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
        gc_fitQA = 0

        # Here granger causality and transfer entropy is calculated from all source and target signals simultaneously. 
        F_value, p_value, best_time_lag_samples = self.granger_causality(target_signal_pp, source_signal_pp, max_time_lag_seconds, 'bic', dt, downsampling_factor, verbose=verbose)
        transfer_entropy_value = self.pin_transfer_entropy(target_signal_pp.T, source_signal_pp.T, best_time_lag_samples)
        latency = best_time_lag_samples * dt * downsampling_factor # At timesteps of dt * downsampling factor

        # Get representative values
        GrCaus_information = np.log2(F_value)
        GrCaus_p = p_value
        transfer_entropy = transfer_entropy_value
        GrCaus_latency = latency
        MeanGrCaus_fitQA = 1

        # for each source signal, calculate all target signals.
        for pre_idx in pre_idx_array:
            for post_idx in post_idx_array:

                # The model order is the maximum number of lagged observations included in the model
                _source, _target = source_signal_pp[:,pre_idx], target_signal_pp[:,post_idx]
                signals = np.vstack([_target, _source]).T

                pairwise_gc_dict = gc_test(signals, [best_time_lag_samples], verbose=False)

                # transfer_entropy_value = self.pin_transfer_entropy(_target, _source, best_time_lag_samples)
                if return_only_infomatrix is False:
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
        
        # Magnitude in base 2 log. If error distribution is gaussian, can be interpreted as bits
        # Multiplied with sample frequency, gives information transfer rate. Lionell Barnett, personal communication:
        # "Note that the log of the F-statistic for the nested VAR model may be interpreted as (Shannon) information 
        # measured in bits (or nats, if natural logarithms are used). The information transfer rate -- a more meaningful 
        # measure -- is then log(F) x f, with unitsof  bits (or nats) per second, where f is the sample frequency in Hz."
        gc_matrix_np_Info = np.log2(gc_matrix_np_F)  

        if return_only_infomatrix is True:
            return gc_matrix_np_Info

        MeanGrCaus_fitQA = gc_fitQA / (pre_idx_array.size * post_idx_array.size)

        # Return one value per analysis (mean of best matching units), indicating GrCaus relation
        return GrCaus_information, GrCaus_p, GrCaus_latency, target_signal_entropy, transfer_entropy, MeanGrCaus_fitQA

    def _analyze_coherence(self, data_dict, source_signal, dt, target_group, t_idx_start=0, t_idx_end=-1):
        '''
        Median sum of coherence spectrum between source and target of correctly classified units
        Assuming that the correctly classified units are on the diagonal indexes [0->0, 1->1 ..., n->n]
        '''
        
        vm_unit = self._get_vm_by_interval(data_dict, target_group, t_idx_start=0, t_idx_end=-1) 
        target_signal = vm_unit / vm_unit.get_best_unit() 

        nsamples = self._get_nsamples(data_dict) 
        nperseg = nsamples//6 
        samp_freq = 1.0 / dt  

        # Cut to requested length 
        source_signal = source_signal[t_idx_start:t_idx_end,:]
        target_signal = target_signal[t_idx_start:t_idx_end,:]

        # Init data matrix
        coherence_matrix_np_sum = np.full_like(np.empty((source_signal.shape[1], target_signal.shape[1])),0)
        coherence_matrix_np_latency = np.full_like(np.empty((source_signal.shape[1], target_signal.shape[1])),0)
        coherence_matrix_np_error = np.full_like(np.empty((source_signal.shape[1], target_signal.shape[1])),0)
        coherence_matrix_np_errorShifted = np.full_like(np.empty((source_signal.shape[1], target_signal.shape[1])),0)

        # Loop units. For each source signal, calculate all target signals.
        pre_idx_array =  np.arange(source_signal.shape[1])
        post_idx_array =  np.arange(target_signal.shape[1])

        for pre_idx in pre_idx_array:
            for post_idx in post_idx_array:
                _source, _target = source_signal[:,pre_idx], target_signal[:,post_idx]

                f, Cxy, Pwelch_spec_x, Pwelch_spec_y, Pxy, lags, corr, coherence_sum, _source_scaled, _target_scaled = \
                            self.get_coherence_of_two_signals(_source, _target, samp_freq=samp_freq, nperseg=nperseg)
                shift_in_seconds = self._get_cross_corr_latency(lags, corr, dt)
                error = self._get_error(_source_scaled, _target_scaled, dt, shift_in_seconds=0)
                errorShifted = self._get_error(_source_scaled, _target_scaled, dt, shift_in_seconds=shift_in_seconds)

                coherence_matrix_np_sum[pre_idx, post_idx] = coherence_sum
                coherence_matrix_np_latency[pre_idx, post_idx] = shift_in_seconds
                coherence_matrix_np_error[pre_idx, post_idx] = error
                coherence_matrix_np_errorShifted[pre_idx, post_idx] = errorShifted

        # Index to diagonal, ie "correctly" classified units and get median of values on diagonal 
        eye_idx = np.eye(source_signal.shape[1], target_signal.shape[1], dtype=bool)
        coherences_on_diagonal = coherence_matrix_np_sum[eye_idx] 
        MedianCoherenceSum = np.nanmedian(coherences_on_diagonal)
        latencies_on_diagonal = coherence_matrix_np_latency[eye_idx] 
        MedianCoherenceLatency = np.nanmedian(latencies_on_diagonal)
        error_on_diagonal = coherence_matrix_np_error[eye_idx] 
        MedianError = np.nanmedian(error_on_diagonal)
        errorShifted_on_diagonal = coherence_matrix_np_errorShifted[eye_idx] 
        MedianErrorShifted = np.nanmedian(errorShifted_on_diagonal)

        if 1:
            print(f'coherence_matrix_np_sum = {coherence_matrix_np_sum}')
            print(f'coherence_matrix_np_latency = {coherence_matrix_np_latency}')
            print(f'coherence_matrix_np_error = {coherence_matrix_np_error}')
            print(f'coherence_matrix_np_errorShifted = {coherence_matrix_np_errorShifted}')

        return MedianCoherenceSum, MedianCoherenceLatency, MedianError, MedianErrorShifted

    def _get_cross_corr_latency(self, lags, corr, dt):
        
        idx = lags>=0
        corr_on_positive_lags = corr[idx]
        positive_lags = lags[idx]
        idx2 = np.argmax(corr_on_positive_lags)
        shift_in_samples = positive_lags[idx2]
        shift_in_seconds = shift_in_samples * dt

        return shift_in_seconds

    def _get_spikes_with_leak(self, spikes, Lambda, dt):

        spikes_leak = np.zeros(spikes.shape)
        n_time_points = spikes.shape[0]

        # Filtered spike train from spikes
        time_vector = np.arange(n_time_points)
        for t in time_vector[1:]:
            spikes_leak[t-1,:] += spikes[t-1,:]

            # the filtered spike train has a leak time constant of Lambda (time points)
            spikes_leak[t,:] = (1 - Lambda * dt) * spikes_leak[t-1,:] 
        
        return spikes_leak 

    def _get_input_with_leak(self, Input, Lambda, dt):
        # Get target output by leaky integration of the input %%
        # for t=2:TimeT
        #     xT(:,t) = (1 - lambda * dt) * xT(:, t-1) + dt * Input(:, t-1); 
        # end
        input_leak = np.zeros(Input.shape)
        n_time_points = Input.shape[0]

        # Filtered input
        time_vector = np.arange(n_time_points)
        for t in time_vector[1:]:
            # the filtered input has a leak time constant of Lambda (time points)
            input_leak[t,:] = (1 - Lambda * dt) * input_leak[t-1,:] + dt * Input[t-1,:]
        
        return input_leak

    def _check_readout_group(self, simulation_engine, readout_group):

        if simulation_engine.lower() in ['cxsystem']:
            if readout_group.lower() in ['e', 'excitatory']:
                readout_group = 'NG1'
            elif readout_group.lower() in ['i', 'inhibitory']:
                readout_group = 'NG2'
            elif 'NG1' not in readout_group and 'NG2' not in readout_group:
                raise ValueError(f'{readout_group} is not valid readout_group for simulation_engine {simulation_engine}, aborting...')
        elif simulation_engine.lower() in ['matlab']:
            if readout_group.lower() in ['ng1','excitatory']:
                readout_group = 'E'
            elif readout_group.lower() in ['ng2', 'inhibitory']:
                readout_group = 'I'
            elif 'E' not in readout_group and 'I' not in readout_group:
                raise ValueError(f'{readout_group} is not valid readout_group for simulation_engine {simulation_engine}, aborting...')
        else:
            raise ValueError(f'{simulation_engine} is not a valid simulation_engine, aborting...')

        return readout_group

    def _get_optimal_decoders(self, target_output, spikes_leak, decoding_method):
        
        # Computing optimal decoders
        # Decs = (spikes_leak' \ target_output')'; % target_output is the input smoothed with membrane leak; Note: (spikes_leak' \ target_output')' = target_output / spikes_leak 
        # target_output = Decs * spikes_leak which can be solved for Decs: xb = a: solve b.T x.T = a.T instead 
        # b: spikes_leak; a: target_output; x: Decs
        # From numpy for matlab users (numpy site)
        # Matlab: b/a; Python: Solve a.T x.T = b.T instead, which is a solution of x a = b for x

        if decoding_method in ['least_squares', 'lstsq']:
            Decs = np.linalg.lstsq(spikes_leak, target_output)[0]
        elif decoding_method in ['pseudoinverse', 'pinv']:
            Decs = np.dot(target_output.T, np.linalg.pinv(spikes_leak.T)) 
            Decs = Decs.T
        else:
            raise NotImplementedError('Unknown method for decoding, try "least_squares", aborting...')

        return Decs

    def get_normalized_error_variance(self, target_output, output):

        # Get MSE error according to Brendel_2020_PLoSComputBiol
        # we compute the variance of the error normalized by the variance of the target population
        Error = np.sum(np.var(target_output - output, axis=0)) / np.sum(np.var(target_output,axis=0))

        return Error

    def get_MSE(self, results_filename=None, data_dict=None, simulation_engine='CxSystem', readout_group='NG1', 
                decoding_method='least_squares', output_type = 'estimated'):
        '''
        Get decoding error. Allows both direct call by default method on single files and with data dictionary for array analysis.
        '''
        
        #  Get input. This is always the project analog input file
        Input =  self.read_input_matfile(filename=self.input_filename, variable='stimulus')
        n_input_time_points = Input.shape[0]

        # Check readoutgroup name, standardize
        readout_group = self._check_readout_group(simulation_engine, readout_group)


        # Get filtered spike train from simulation. 
        if 'matlab' in simulation_engine.lower():
            if data_dict is None:
                assert results_filename is not None, 'For matlab, you need to provide the workspace data as "results_filename", aborting...'
                # Input scaling factor A is 2000 for matlab results
                data_dict = self.getData(filename=results_filename)
            target_rO_name = f'rO{readout_group}'
            spikes_leak = data_dict[target_rO_name].T
            Lambda = data_dict['lambda']
            dt = data_dict['dt']
            n_time_points = spikes_leak.shape[0]

        elif 'cxsystem' in simulation_engine.lower():
            # Input scaling factor A is 15 for python results
            if data_dict is None:
                data_dict = self.getData(filename=results_filename, data_type='results')
            n_time_points = self._get_nsamples(data_dict)
            NG_name = [n for n in data_dict['spikes_all'].keys() if f'{readout_group}' in n ][0]
            n_neurons = data_dict['Neuron_Groups_Parameters'][NG_name]['number_of_neurons']

            # Get dt
            dt = self._get_dt(data_dict)

            # Get Lambda, a.k.a. tau_soma, but in time points. Can be float.
            Lambda_unit = self._get_namespace_variable(data_dict, readout_group, variable_name='taum_soma')
            Lambda = Lambda_unit.base / dt

            # Get spikes
            spike_idx = data_dict['spikes_all'][NG_name]['i']
            spike_times = data_dict['spikes_all'][NG_name]['t']
            spike_times_idx = np.array(spike_times / (dt * b2u.second) , dtype=int)

            # Create spike vector 
            spikes = np.zeros([n_time_points, n_neurons])
            spikes[spike_times_idx, spike_idx] = 1
            # Spikes with leak
            spikes_leak = self._get_spikes_with_leak(spikes, Lambda, dt)

        # Get input with leak, a.k.a. the target output 
        input_leak = self._get_input_with_leak(Input, Lambda, dt)
        target_output = input_leak

        # Get output
        assert output_type in ['estimated', 'simulated'], 'Unknown output type, should be "estimated" or "simulated", aborting...'
        if output_type == 'estimated':
            # Get optimal decoders with analytical method
            Decs = self. _get_optimal_decoders(target_output, spikes_leak, decoding_method)
            # Get estimated output
            estimated_output = np.dot(Decs.T, spikes_leak.T)
            output = estimated_output.T
        elif output_type == 'simulated':
            # Get simulated vm values for target group. Only valid for cxsystem data. 
            simulated_output_vm_unit = self._get_vm_by_interval(data_dict, NG=self.NG_name, t_idx_start=0, t_idx_end=None)
            simulated_output_vm = simulated_output_vm_unit / simulated_output_vm_unit.get_best_unit()
            # both output and target data are scaled to -1,1 for comparison. Minmax keeps distribution histogram form intact.
            simulated_output = self.scaler(simulated_output_vm, scale_type='minmax')
            output = simulated_output
            target_output = self.scaler(target_output, scale_type='minmax')

        Error = self.get_normalized_error_variance(target_output, output)

        assert n_input_time_points == n_time_points, 'input and simulation data time vector lengths do not match, aborting...'

        return Error, target_output, output

    def get_PCA(self, data, n_components=2, col_names=None, extra_points=None, extra_points_at_edge_of_gamut=False):

        values_np_scaled = self.scaler(data, scale_type='standard') # scales to zero mean, sd of one

        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(values_np_scaled)

        if col_names is not None:
            index = col_names
        else:
            index = list(range(n_components))
            
        principal_axes_in_PC_space = pd.DataFrame(pca.components_.T, columns=[f'PC{pc}' for pc in range(n_components)], index=index)
        explained_variance_ratio = pca.explained_variance_ratio_

        if extra_points is not None:
            pca_extra_points = pca.transform(extra_points)
        else:
            pca_extra_points = None

        if extra_points_at_edge_of_gamut is True:
            xmin, xmax, ymin, ymax = np.min(pca_data[:,0]), np.max(pca_data[:,0]), np.min(pca_data[:,1]), np.max(pca_data[:,1])
            extra_points_np = np.zeros(pca_extra_points.shape)
            for rowidx, row in  enumerate(pca_extra_points):
                x0,y0=zip(row)
                x0,y0=x0[0],y0[0] # zip provides tuples, must get the floats

                xyratio = np.abs(x0/y0)
                if x0 < 0 and y0 >= 0:
                    xymaxratio = np.abs(xmin/ymax)
                    if xyratio >= xymaxratio:
                        xe = xmin
                        c = np.abs(xmin/x0)
                        ye = c * y0
                    else:
                        ye = ymax
                        c = np.abs(ymax/y0)
                        xe = c * x0
                if x0 >= 0 and y0 >= 0:
                    xymaxratio = np.abs(xmax/ymax)
                    if xyratio >= xymaxratio:
                        xe = xmax
                        c = np.abs(xmax/x0)
                        ye = c * y0
                    else:
                        ye = ymax
                        c = np.abs(ymax/y0)
                        xe = c * x0
                if x0 < 0 and y0 < 0:
                    xymaxratio = np.abs(xmin/ymin)
                    if xyratio >= xymaxratio:
                        xe = xmin
                        c = np.abs(xmin/x0)
                        ye = c * y0
                    else:
                        ye = ymin
                        c = np.abs(ymin/y0)
                        xe = c * x0
                if x0 >= 0 and y0 < 0:
                    xymaxratio = np.abs(xmax/ymin)
                    if xyratio >= xymaxratio:
                        xe = xmax
                        c = np.abs(xmax/x0)
                        ye = c * y0
                    else:
                        ye = ymin
                        c = np.abs(ymin/y0)
                        xe = c * x0
                extra_points_np[rowidx,:] = np.array([xe, ye])
            pca_extra_points = extra_points_np

        return pca_data, principal_axes_in_PC_space, explained_variance_ratio, pca_extra_points

    def _analyze_meanerror(self, data_dict, **kwargs):
        
        decoding_method = kwargs['decoding_method'] 
        simulation_engine='CxSystem'

        MeanEstimErr_E, target_output, output = self.get_MSE( data_dict=data_dict, simulation_engine=simulation_engine, 
                                                readout_group='excitatory', decoding_method=decoding_method, output_type='estimated')
        MeanEstimErr_I, target_output, output = self.get_MSE( data_dict=data_dict, simulation_engine=simulation_engine, 
                                                readout_group='inhibitory', decoding_method=decoding_method, output_type='estimated')
        MeanSimErr_O, target_output, output = self.get_MSE(data_dict=data_dict, output_type='simulated')

        return MeanEstimErr_E, MeanEstimErr_I, MeanSimErr_O

    def _analyze_classification_performance(self, y_data):
        # eye_idx = np.eye(source_signal.shape[1], target_signal.shape[1], dtype=bool)
        eye_idx = np.eye(y_data.shape[0], y_data.shape[1], dtype=bool)
        y_true = eye_idx

        y_pred = np.zeros(y_data.shape)
        row_idx = y_data.argmax(axis=0)
        column_idx = y_data.argmax(axis=1)
        y_pred[row_idx, column_idx] = 1

        Accuracy = accuracy_score(y_true, y_pred)

        return Accuracy

    def get_analyzed_array_as_df(self, data_df, analysisHR=None, t_idx_start=0, t_idx_end=None, **kwargs):
        '''
        Call necessary analysis and build dataframe
        '''
    
        # Get neuron group names
        filename_0 = data_df['Full path'].values[0]
        data = self.getData(filename_0)
        NG_list = [n for n in data[self.map_data_types[analysisHR.lower()]].keys() if 'NG' in n]
        target_group = self.NG_name            

        # Add neuron group columns
        if analysisHR.lower() in ['meanfr', 'meanvm', 'eicurrentdiff']:
            for NG in NG_list:
                data_df[f'{analysisHR}_' + NG] = np.nan
        elif analysisHR.lower() in ['coherence']:
            data_df[f'{analysisHR}_' + target_group + '_Sum'] = np.nan            
            data_df[f'{analysisHR}_' + target_group + '_Latency'] = np.nan            
            data_df[f'{analysisHR}_' + target_group + '_error'] = np.nan            
            data_df[f'{analysisHR}_' + target_group + '_errorShifted'] = np.nan            
        elif analysisHR.lower() in ['grcaus']:
            data_df[f'{analysisHR}_' + target_group + '_Information'] = np.nan
            data_df[f'{analysisHR}_' + target_group + '_p'] = np.nan 
            data_df[f'{analysisHR}_' + target_group + '_TransfEntropy'] = np.nan 
            data_df[f'{analysisHR}_' + target_group + '_latency'] = np.nan
            data_df[f'{analysisHR}_' + target_group + '_targetEntropy'] = np.nan
            data_df[f'{analysisHR}_' + target_group + '_fitQuality'] = np.nan
        elif analysisHR.lower() in ['meanerror']:
            data_df[f'{analysisHR}_' + '_ExcErr'] = np.nan
            data_df[f'{analysisHR}_' + '_InhErr'] = np.nan
            data_df[f'{analysisHR}_' + '_SimErr'] = np.nan
        elif analysisHR.lower() in ['classify']:
            data_df[f'{analysisHR}_' + 'Accuracy'] = np.nan

        target_signal_dt = self._get_dt(data)

        if analysisHR.lower() in ['coherence', 'grcaus', 'classify']:
            # Get reference data for granger causality and coherence analyses
            analog_input = self.getData( self.input_filename, data_type=None)
            source_signal = analog_input['stimulus'].T # We want time x units
            source_signal_dt = analog_input['frameduration'] / 1000 # assuming input dt in milliseconds

            assert target_signal_dt == source_signal_dt, \
                'Different sampling rates in input and output has not been implemented, aborting...'

        # Get duration
        if t_idx_end is None:
            t_idx_end = int(data['runtime']  / target_signal_dt)

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
            elif analysisHR.lower() in ['coherence']:
                MedianCoherenceSum, MedianCoherenceLatency, MedianError, MedianErrorShifted = \
                    self._analyze_coherence(data, source_signal, target_signal_dt, target_group, \
                    t_idx_start=t_idx_start, t_idx_end=t_idx_end)
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_Sum'] = MedianCoherenceSum
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_Latency'] = MedianCoherenceLatency
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_error'] = MedianError
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_errorShifted'] = MedianErrorShifted
            elif analysisHR.lower() in ['grcaus']: # Information transfer
                GrCaus_information, GrCaus_p, GrCaus_latency, target_entropy, transfer_entropy, MeanGrCaus_fitQA = \
                    self._analyze_grcaus(data, source_signal, target_signal_dt, target_group, t_idx_start=t_idx_start, 
                    t_idx_end=t_idx_end, **kwargs)
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_Information'] = GrCaus_information
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_p'] = GrCaus_p
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_TransfEntropy'] = transfer_entropy
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_latency'] = GrCaus_latency
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_targetEntropy'] = target_entropy
                data_df.loc[this_index,f'{analysisHR}_' + target_group + '_fitQuality'] = MeanGrCaus_fitQA
            elif analysisHR.lower() in ['meanerror']: # Reconstruction error
                MeanEstimErr_E, MeanEstimErr_I, MeanSimErr_O = self._analyze_meanerror(data, **kwargs)
                data_df.loc[this_index,f'{analysisHR}_' + '_ExcErr'] = MeanEstimErr_E
                data_df.loc[this_index,f'{analysisHR}_' + '_InhErr'] = MeanEstimErr_I 
                data_df.loc[this_index,f'{analysisHR}_' + '_SimErr'] = MeanSimErr_O 
            elif analysisHR.lower() in ['classify']: # According to sensory information
                gc_matrix_np_Info = self._analyze_grcaus(data, source_signal, target_signal_dt, target_group, verbose=False,  
                    return_only_infomatrix=True, t_idx_start=t_idx_start, t_idx_end=t_idx_end, **kwargs)
                Accuracy = self._analyze_classification_performance(gc_matrix_np_Info)
                data_df.loc[this_index,f'{analysisHR}_' + 'Accuracy'] = Accuracy

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

        # Init txt file for Granger causality fit quality diagnostics
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

        analyzed_data_df.to_csv(csv_name_out, index=False)

    def analyze_plasticity(self, n_iter=1): # UNDER CONSTRUCTION
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

    # analysis.show_readout_on_input(NG_name, normalize=False, filename='Replica_test_results_20210114_1750000.gz')    
    # analysis.show_spikes(filename='Replica_test_results_20210114_1750000.gz')
    
    # plt.show()
    pass