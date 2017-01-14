// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/MasquelierSTDP.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, MasquelierSTDP);

namespace Backend {
  namespace CUDA {
    MasquelierSTDP::~MasquelierSTDP() {
      CudaSafeCall(cudaFree(index_of_last_afferent_synapse_to_spike));
      CudaSafeCall(cudaFree(isindexed_ltd_synapse_spike));
      CudaSafeCall(cudaFree(index_of_first_synapse_spiked_after_postneuron));
    }

    void MasquelierSTDP::reset_state() {
      STDP::reset_state();

      CudaSafeCall(cudaMemcpy((void*)index_of_last_afferent_synapse_to_spike,
                              (void*)frontend()->index_of_last_afferent_synapse_to_spike,
                              sizeof(int)*frontend()->neurs->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)isindexed_ltd_synapse_spike,
                              (void*)frontend()->isindexed_ltd_synapse_spike,
                              sizeof(bool)*frontend()->neurs->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)index_of_first_synapse_spiked_after_postneuron,
                              (void*)frontend()->index_of_first_synapse_spiked_after_postneuron,
                              sizeof(int)*frontend()->neurs->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
    }

    void MasquelierSTDP::prepare() {
      STDP::prepare();

      allocate_device_pointers();
    }

    void MasquelierSTDP::allocate_device_pointers() {
      // The following doesn't do anything in original code...
      // ::Backend::CUDA::STDP::allocate_device_pointers();

      CudaSafeCall(cudaMalloc((void **)&index_of_last_afferent_synapse_to_spike, sizeof(int)*frontend()->neurs->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&isindexed_ltd_synapse_spike, sizeof(int)*frontend()->neurs->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&index_of_first_synapse_spiked_after_postneuron, sizeof(int)*frontend()->neurs->total_number_of_neurons));
    }

    void MasquelierSTDP::apply_stdp_to_synapse_weights(float current_time_in_seconds) {
      // First reset the indices array
      // In order to carry out nearest spike potentiation only, we must find the spike arriving at each neuron which has the smallest time diff
    get_indices_to_apply_stdp<<<neurons_backend->number_of_neuron_blocks_per_grid, synapses_backend->threads_per_block>>>
      (synapses_backend->postsynaptic_neuron_indices,
       neurons_backend->last_spike_time_of_each_neuron,
       synapses_backend->stdp,
       synapses_backend->time_of_last_spike_to_reach_synapse,
       index_of_last_afferent_synapse_to_spike,
       isindexed_ltd_synapse_spike,
       index_of_first_synapse_spiked_after_postneuron,
       current_time_in_seconds,
       frontend()->syns->total_number_of_synapses);
    CudaCheckError();

    apply_stdp_to_synapse_weights_kernel<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
      (synapses_backend->postsynaptic_neuron_indices,
       neurons_backend->last_spike_time_of_each_neuron,
       synapses_backend->stdp,
       synapses_backend->time_of_last_spike_to_reach_synapse,
       synapses_backend->synaptic_efficacies_or_weights,
       index_of_last_afferent_synapse_to_spike,
       isindexed_ltd_synapse_spike,
       index_of_first_synapse_spiked_after_postneuron,
       *(frontend()->stdp_params),
       current_time_in_seconds,
       frontend()->neurs->total_number_of_neurons);
    CudaCheckError();
    }

    // Find nearest spike
    __global__ void apply_stdp_to_synapse_weights_kernel
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     struct masquelier_stdp_parameters_struct stdp_vars,
     float currtime,
     size_t total_number_of_post_neurons){
      // Global Index
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all synapses
      while (idx < total_number_of_post_neurons) {
        // Get the synapse on which to do LTP/LTD
        int index_of_LTP_synapse = d_index_of_last_afferent_synapse_to_spike[idx];
        int index_of_LTD_synapse = d_index_of_first_synapse_spiked_after_postneuron[idx];

        // If we are to carry out STDP on LTP synapse
        if(d_stdp[index_of_LTP_synapse]){
          float last_syn_spike_time = d_time_of_last_spike_to_reach_synapse[index_of_LTP_synapse];
          float last_neuron_spike_time = d_last_spike_time_of_each_neuron[idx];
          float new_syn_weight = d_synaptic_efficacies_or_weights[index_of_LTP_synapse];

          if (last_neuron_spike_time == currtime){
            float diff = currtime - last_syn_spike_time;
            // Only carry out LTP if the difference is greater than some range
            if (diff < 7*stdp_vars.tau_plus && diff > 0){
              float weightchange = stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus);
              // Update weights
              new_syn_weight += weightchange;
              // Ensure that the weights are clipped to 1.0f
              new_syn_weight = min(new_syn_weight, 1.0f);
            }
          }
          // Update the synaptic weight as required
          d_synaptic_efficacies_or_weights[index_of_LTP_synapse] = new_syn_weight;
        }

        // Get the synapse for LTD
        if (d_isindexed_ltd_synapse_spike[idx]){
          if (index_of_LTD_synapse >= 0){
            if (d_stdp[index_of_LTD_synapse]){

              float last_syn_spike_time = d_time_of_last_spike_to_reach_synapse[index_of_LTD_synapse];
              float last_neuron_spike_time = d_last_spike_time_of_each_neuron[idx];
              float new_syn_weight = d_synaptic_efficacies_or_weights[index_of_LTD_synapse];

              // Set the index to negative (i.e. Reset it)
              d_index_of_first_synapse_spiked_after_postneuron[idx] = -1;

              float diff = last_syn_spike_time - last_neuron_spike_time;
              // Only carry out LTD if the difference is in some range
              if (diff < 7*stdp_vars.tau_minus && diff > 0){
                float weightchange = stdp_vars.a_minus * expf(-diff / stdp_vars.tau_minus);
                // Update the weights
                new_syn_weight -= weightchange;
                // Ensure that the weights are clipped to 0.0f
                new_syn_weight = max(new_syn_weight, 0.0f);
              }
              d_synaptic_efficacies_or_weights[index_of_LTD_synapse] = new_syn_weight;
            }
          }	
        }
        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }

    __global__ void get_indices_to_apply_stdp
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_time_of_last_spike_to_reach_synapse,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     float currtime,
     size_t total_number_of_synapse){
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running through all neurons:
      while (idx < total_number_of_synapse){
        int postsynaptic_neuron = d_postsyns[idx];

        // Check whether a synapse reached a neuron this timestep
        if (d_time_of_last_spike_to_reach_synapse[idx] == currtime){
          // Atomic Exchange the new synapse index
          atomicExch(&d_index_of_last_afferent_synapse_to_spike[postsynaptic_neuron], idx);
        }
		
        // Check (if we need to) whether a synapse has fired
        if (d_isindexed_ltd_synapse_spike[postsynaptic_neuron]){
          if (d_time_of_last_spike_to_reach_synapse[idx] == currtime){
            d_isindexed_ltd_synapse_spike[postsynaptic_neuron] = true;
            atomicExch(&d_index_of_first_synapse_spiked_after_postneuron[postsynaptic_neuron], idx);
          }
        }
        // Check whether a neuron has fired
        if (d_last_spike_time_of_each_neuron[postsynaptic_neuron] == currtime){
          d_isindexed_ltd_synapse_spike[postsynaptic_neuron] = false;
        }
        // Increment index
        idx += blockDim.x * gridDim.x;
      }
    }
  }
}
