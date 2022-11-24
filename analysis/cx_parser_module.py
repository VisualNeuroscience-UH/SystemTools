import brian2.units as b2u

import numpy as np

class CXParser():

    def get_dt(self, data):
        dt = (data['time_vector'][1] - data['time_vector'][0]) / b2u.second
        return dt

    def get_n_samples(self, data):
        nsamples = len(data['time_vector'])
        assert nsamples == data["time_vector"].shape[0], 'time_vector shape inconsistency, aborting...'
        return nsamples

    def get_namespace_variable(self, data, readout_group, variable_name='taum_soma'):

        NG_name = [n for n in data['Neuron_Groups_Parameters'].keys() if f'{readout_group}' in n ][0]
        variable_value_unit = data['Neuron_Groups_Parameters'][NG_name]['namespace'][variable_name]

        return variable_value_unit

    def get_vm_by_interval(self, data, NG=None, t_idx_start=0, t_idx_end=None):

        if NG is None:
            vm = data['vm']  # data_by_group already
        else:
            vm = data['vm_all'][NG]['vm']

        n_samples = vm.shape[0]

        # the idx end is correct, but slicing below would drop one sample
        if t_idx_end is None:
            t_idx_end = n_samples
        elif t_idx_end < 0:
            t_idx_end = n_samples + t_idx_end
    
        return vm[t_idx_start:t_idx_end, :]

    def get_currents_by_interval(self, data, NG, t_idx_start=0, t_idx_end=None):

        ge = data['ge_soma_all'][NG]['ge_soma']
        gi = data['gi_soma_all'][NG]['gi_soma']
        vm = data['vm_all'][NG]['vm']

        # Get necessary variables
        # Calculate excitatory, inhibitory (and leak) currents
        gl = data['Neuron_Groups_Parameters'][NG]['namespace']['gL']
        El = data['Neuron_Groups_Parameters'][NG]['namespace']['EL']
        I_leak = gl * (El - vm)

        # If no driving force in neuron vm model synapses. This is currently denoted in neuron group name by _CI_ prefix
        if '_CI_' in NG:
            I_e = ge * b2u.mV
            I_i = gi * b2u.mV
        else:
            Ee = data['Neuron_Groups_Parameters'][NG]['namespace']['Ee']
            Ei = data['Neuron_Groups_Parameters'][NG]['namespace']['Ei']
            I_e = ge * (Ee - vm)
            I_i = gi * (Ei - vm)

        return I_e[t_idx_start:t_idx_end, :], I_i[t_idx_start:t_idx_end, :], I_leak[t_idx_start:t_idx_end, :]

    def _get_spikes_by_interval(self, data, NG, t_idx_start, t_idx_end):

        data_by_group = data["spikes_all"][NG]

        # Get and mark MeanFR to df
        N_neurons = data_by_group["count"].size

        # spikes by interval needs seconds, thus we need to multiply with dt
        dt = self.get_dt(data)

        t_start=t_idx_start * dt
        t_end=t_idx_end * dt

        spikes = data_by_group["t"][
            np.logical_and(
                data_by_group["t"] > t_start * b2u.second,
                data_by_group["t"] < t_end * b2u.second,
            )
        ]

        return N_neurons, spikes, dt

