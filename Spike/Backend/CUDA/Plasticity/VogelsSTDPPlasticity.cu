// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/VogelsSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, VogelsSTDPPlasticity);

namespace Backend {
  namespace CUDA {
    VogelsSTDPPlasticity::~VogelsSTDPPlasticity() {
      CudaSafeCall(cudaFree(vogels_pre_memory_trace));
      CudaSafeCall(cudaFree(vogels_post_memory_trace));
    }

    void VogelsSTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();

      CudaSafeCall(cudaMemcpy((void*)vogels_pre_memory_trace,
                              (void*)vogels_memory_trace_reset,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)vogels_post_memory_trace,
                              (void*)vogels_memory_trace_reset,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
    }

    void VogelsSTDPPlasticity::prepare() {
      STDPPlasticity::prepare();

      vogels_memory_trace_reset = (float*)malloc(sizeof(float)*total_number_of_plastic_synapses);
      for (int i=0; i < total_number_of_plastic_synapses; i++){
	vogels_memory_trace_reset[i] = 0.0f;
      }

      allocate_device_pointers();
    }

    void VogelsSTDPPlasticity::allocate_device_pointers() {
      // The following doesn't do anything in original code...
      // ::Backend::CUDA::STDPPlasticity::allocate_device_pointers();

      CudaSafeCall(cudaMalloc((void **)&vogels_pre_memory_trace, sizeof(int)*total_number_of_plastic_synapses));
      CudaSafeCall(cudaMalloc((void **)&vogels_post_memory_trace, sizeof(int)*total_number_of_plastic_synapses));
    }

    void VogelsSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {

    // Vogels update rule requires a neuron wise memory trace. This must be updated upon neuron firing.
    vogels_apply_stdp_to_synapse_weights_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
      (synapses_backend->presynaptic_neuron_indices,
       synapses_backend->postsynaptic_neuron_indices,
       neurons_backend->last_spike_time_of_each_neuron,
       synapses_backend->synaptic_efficacies_or_weights,
       vogels_pre_memory_trace,
       vogels_post_memory_trace,
       *(frontend()->stdp_params),
       current_time_in_seconds,
       timestep,
       plastic_synapse_indices,
       total_number_of_plastic_synapses);
    CudaCheckError();
    }

    // Find nearest spike
    __global__ void vogels_apply_stdp_to_synapse_weights_kernel
    (int* d_presyns,
     int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     float* d_synaptic_efficacies_or_weights,
     float* vogels_pre_memory_trace,
     float* vogels_post_memory_trace,
     struct vogels_stdp_plasticity_parameters_struct stdp_vars,
     float currtime,
     float timestep,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses){
      // Global Index
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all stdp synapses
      while (indx < total_number_of_plastic_synapses){
	// First decaying the memory traces
	vogels_pre_memory_trace[indx] += - vogels_pre_memory_trace[indx]*(timestep / stdp_vars.tau_istdp);
	vogels_post_memory_trace[indx] += - vogels_post_memory_trace[indx]*(timestep / stdp_vars.tau_istdp);
        // Getting an index for the correct synapse
        int idx = d_plastic_synapse_indices[indx];

        // Find the pre and post synaptic neuron ids:
        int pre_neuron_id = d_presyns[idx];
        int post_neuron_id = d_postsyns[idx];

        // Check whether the pre-synaptic neuron has fired now
        if (d_last_spike_time_of_each_neuron[post_neuron_id] == currtime)
          vogels_post_memory_trace[indx] += 1.0f;
        if (d_last_spike_time_of_each_neuron[pre_neuron_id] == currtime)
          vogels_pre_memory_trace[indx] += 1.0f;

        if (d_last_spike_time_of_each_neuron[pre_neuron_id] == currtime){
          float new_syn_weight = d_synaptic_efficacies_or_weights[idx];
          new_syn_weight += stdp_vars.learningrate*(vogels_post_memory_trace[indx]);
          // Alpha must be calculated as 2 * targetrate * tau_istdp
          new_syn_weight += - stdp_vars.learningrate*(2.0*stdp_vars.targetrate*stdp_vars.tau_istdp);
          if (new_syn_weight < 0.0f)
            new_syn_weight = 0.0f;
          d_synaptic_efficacies_or_weights[idx] = new_syn_weight;
        }

        // Check whether the post-synaptic neuron has fired now
        if (d_last_spike_time_of_each_neuron[post_neuron_id] == currtime){
          float new_syn_weight = d_synaptic_efficacies_or_weights[idx];
          new_syn_weight += stdp_vars.learningrate*(vogels_pre_memory_trace[indx]);
          d_synaptic_efficacies_or_weights[idx] = new_syn_weight;
        }
        indx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }
  }
}
