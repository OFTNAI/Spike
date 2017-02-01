// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/VogelsSTDP.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, VogelsSTDP);

namespace Backend {
  namespace CUDA {
    VogelsSTDP::~VogelsSTDP() {
      CudaSafeCall(cudaFree(vogels_memory_trace));
    }

    void VogelsSTDP::reset_state() {
      STDP::reset_state();

      CudaSafeCall(cudaMemcpy((void*)vogels_memory_trace,
                              (void*)frontend()->vogels_memory_trace,
                              sizeof(float)*frontend()->neurs->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
    }

    void VogelsSTDP::prepare() {
      STDP::prepare();

      allocate_device_pointers();
    }

    void VogelsSTDP::allocate_device_pointers() {
      // The following doesn't do anything in original code...
      // ::Backend::CUDA::STDP::allocate_device_pointers();

      CudaSafeCall(cudaMalloc((void **)&vogels_memory_trace, sizeof(int)*frontend()->neurs->total_number_of_neurons));
    }

    void VogelsSTDP::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {

    // Vogels update rule requires a neuron wise memory trace. This must be updated upon neuron firing.
    vogels_update_memory_trace<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
      (neurons_backend->last_spike_time_of_each_neuron,
       current_time_in_seconds,
       vogels_memory_trace,
       *(frontend()->stdp_params),
       frontend()->neurs->total_number_of_neurons);
    CudaCheckError();

    vogels_apply_stdp_to_synapse_weights_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
      (synapses_backend->presynaptic_neuron_indices,
       synapses_backend->postsynaptic_neuron_indices,
       neurons_backend->last_spike_time_of_each_neuron,
       synapses_backend->stdp,
       synapses_backend->synaptic_efficacies_or_weights,
       vogels_memory_trace,
       *(frontend()->stdp_params),
       current_time_in_seconds,
       timestep,
       stdp_synapse_indices,
       total_number_of_stdp_synapses);
    CudaCheckError();
    }

    // Find nearest spike
    __global__ void vogels_apply_stdp_to_synapse_weights_kernel
    (int* d_presyns,
     int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     bool* d_stdp,
     float* d_synaptic_efficacies_or_weights,
     float* vogels_memory_trace,
     struct vogels_stdp_parameters_struct stdp_vars,
     float currtime,
     float timestep,
     int* d_stdp_synapse_indices,
     size_t total_number_of_stdp_synapses){
      // Global Index
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all stdp synapses
      while (indx < total_number_of_stdp_synapses){
        // Getting an index for the correct synapse
        int idx = d_stdp_synapse_indices[indx];

        // Find the pre and post synaptic neuron ids:
        int pre_neuron_id = d_presyns[idx];
        int post_neuron_id = d_postsyns[idx];

        // Check whether the pre-synaptic neuron has fired now
        if (d_last_spike_time_of_each_neuron[pre_neuron_id] == currtime){
          float new_syn_weight = d_synaptic_efficacies_or_weights[idx];
          new_syn_weight += stdp_vars.learningrate*(vogels_memory_trace[post_neuron_id]);
          // Alpha must be calculated as 2 * targetrate * tau_istdp
          new_syn_weight += stdp_vars.learningrate*(2.0*stdp_vars.targetrate*stdp_vars.tau_istdp);
          d_synaptic_efficacies_or_weights[idx] = new_syn_weight;
        }

        // Check whether the post-synaptic neuron has fired now
        if (d_last_spike_time_of_each_neuron[post_neuron_id] == currtime){
          float new_syn_weight = d_synaptic_efficacies_or_weights[idx];
          new_syn_weight += stdp_vars.learningrate*(vogels_memory_trace[pre_neuron_id]);
          d_synaptic_efficacies_or_weights[idx] = new_syn_weight;
        }
        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }

    __global__ void vogels_update_memory_trace
    (float* d_last_spike_time_of_each_neuron,
     float currtime,
     float* vogels_memory_trace,
     struct vogels_stdp_parameters_struct stdp_vars,
     size_t total_number_of_neurons){
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running through all neurons:
      while (idx < total_number_of_neurons){

        // Decay all neuron memory traces
        vogels_memory_trace[idx] += (- vogels_memory_trace[idx] / stdp_vars.tau_istdp)*timestep;

        // If the neuron has fired, update its memory trace
        if (d_last_spike_time_of_each_neuron[idx] == currtime){
          vogels_memory_trace[idx] += 1.0f;
        }

        // Increment index
        indx += blockDim.x * gridDim.x;
      }
    }
  }
}
