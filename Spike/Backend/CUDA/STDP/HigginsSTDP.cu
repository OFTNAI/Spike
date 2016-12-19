// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/HigginsSTDP.hpp"

namespace Backend {
  namespace CUDA {
    void HigginsSTDP::prepare() {
      STDP::prepare();

      // allocate_device_pointers();
    }

    void HigginsSTDP::reset_state() {
      STDP::reset_state();
    }

    void HigginsSTDP::push_data_front() {
      STDP::push_data_front();
    }

    void HigginsSTDP::pull_data_back() {
      STDP::pull_data_back();
    }

    void HigginsSTDP::apply_ltd_to_synapse_weights(float current_time_in_seconds) {
      izhikevich_apply_ltd_to_synapse_weights_kernel<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
        (synapses_backend->time_of_last_spike_to_reach_synapse,
         synapses_backend->synaptic_efficacies_or_weights,
         synapses_backend->stdp,
         neurons_backend->last_spike_time_of_each_neuron,
         synapses_backend->postsynaptic_neuron_indices,
         current_time_in_seconds,
         *(frontend()->stdp_params), // Should make device copy?
         frontend()->syns->total_number_of_synapses);

      CudaCheckError();
    }

    void HigginsSTDP::apply_ltp_to_synapse_weights(float current_time_in_seconds) {
      izhikevich_apply_ltp_to_synapse_weights_kernel<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
        (synapses_backend->postsynaptic_neuron_indices,
         neurons_backend->last_spike_time_of_each_neuron,
         synapses_backend->stdp,
         synapses_backend->time_of_last_spike_to_reach_synapse,
         synapses_backend->synaptic_efficacies_or_weights,
         *(frontend()->stdp_params),
         current_time_in_seconds,
         frontend()->syns->total_number_of_synapses);

      CudaCheckError();
    }

    // LTP on synapses
    __global__ void izhikevich_apply_ltp_to_synapse_weights_kernel
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     struct higgins_stdp_parameters_struct stdp_vars,
     float currtime,
     size_t total_number_of_synapse) {

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_synapse) {
        // Get the synapses upon which we should do LTP
        // Reversed indexing to check post->pre synapses
        if ((d_last_spike_time_of_each_neuron[d_postsyns[idx]] == currtime) && (d_stdp[idx] == true)){
          // Get the last active time / weight of the synapse
          // Calc time difference and weight change
          float diff = currtime - d_time_of_last_spike_to_reach_synapse[idx];
          float weightchange = (stdp_vars.w_max - d_synaptic_efficacies_or_weights[idx]) * (stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus));
          // Update weights
          d_synaptic_efficacies_or_weights[idx] += weightchange;
        }
        idx += blockDim.x * gridDim.x;

      }
    }

    // LTD on Synapses
    __global__ void izhikevich_apply_ltd_to_synapse_weights_kernel
    (float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     bool* d_stdp,
     float* d_last_spike_time_of_each_neuron,
     int* d_postsyns,
     float currtime,
     struct higgins_stdp_parameters_struct stdp_vars,
     size_t total_number_of_synapse){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_synapse) {

        // Get the locations for updating
        // Get the synapses that are to be LTD'd
        if ((d_time_of_last_spike_to_reach_synapse[idx] == currtime) && (d_stdp[idx] == 1)) {
          float diff = d_last_spike_time_of_each_neuron[d_postsyns[idx]] - currtime;
          // STDP Update Rule
          float weightscale = stdp_vars.w_max * stdp_vars.a_minus * expf(diff / stdp_vars.tau_minus);
          // Now scale the weight (using an inverted column/row)
          d_synaptic_efficacies_or_weights[idx] += weightscale; 
        }
        idx += blockDim.x * gridDim.x;
      }
    }

  }
}
