// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/vanRossumSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, vanRossumSTDPPlasticity);

namespace Backend {
  namespace CUDA {
    vanRossumSTDPPlasticity::~vanRossumSTDPPlasticity() {
      CudaSafeCall(cudaFree(index_of_last_afferent_synapse_to_spike));
      CudaSafeCall(cudaFree(isindexed_ltd_synapse_spike));
      CudaSafeCall(cudaFree(index_of_first_synapse_spiked_after_postneuron));
      CudaSafeCall(cudaFree(stdp_pre_memory_trace));
      CudaSafeCall(cudaFree(stdp_post_memory_trace));
      free(h_stdp_trace);
    }

    void vanRossumSTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();
      
      CudaSafeCall(cudaMemcpy((void*)stdp_pre_memory_trace,
                              (void*)h_stdp_trace,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)stdp_post_memory_trace,
                              (void*)h_stdp_trace,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
    }

    void vanRossumSTDPPlasticity::prepare() {
      STDPPlasticity::prepare();

      allocate_device_pointers();
    }

    void vanRossumSTDPPlasticity::allocate_device_pointers() {
      // The following doesn't do anything in original code...
      // ::Backend::CUDA::STDPPlasticity::allocate_device_pointers();
      CudaSafeCall(cudaMalloc((void **)&stdp_pre_memory_trace, sizeof(float)*total_number_of_plastic_synapses));
      CudaSafeCall(cudaMalloc((void **)&stdp_post_memory_trace, sizeof(float)*total_number_of_plastic_synapses));
      // Also allocate and set trace template
      h_stdp_trace = (float *)malloc( sizeof(float) * total_number_of_plastic_synapses);
      for (int i=0; i < total_number_of_plastic_synapses; i++){
        h_stdp_trace[i] = 0.0f;
      }
    }

    void vanRossumSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
      if (total_number_of_plastic_synapses > 0){ 
        vanrossum_ltp_and_ltd<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
          (synapses_backend->postsynaptic_neuron_indices,
           synapses_backend->time_of_last_spike_to_reach_synapse,
           neurons_backend->last_spike_time_of_each_neuron,
           synapses_backend->synaptic_efficacies_or_weights,
           stdp_pre_memory_trace,
           stdp_post_memory_trace,
           *(frontend()->stdp_params),
           timestep,
	   frontend()->model->timestep_grouping,
           current_time_in_seconds,
           plastic_synapse_indices,
           total_number_of_plastic_synapses);
        
      }
    }

    // ALL SPIKE IMPLEMENTATION
    __global__ void vanrossum_ltp_and_ltd
          (int* d_postsyns,
           float* d_time_of_last_spike_to_reach_synapse,
           float* d_last_spike_time_of_each_neuron,
           float* d_synaptic_efficacies_or_weights,
           float* stdp_pre_memory_trace,
           float* stdp_post_memory_trace,
           struct vanrossum_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
	   int timestep_grouping,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses){
      // Global Index
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all neurons
      while (indx < total_number_of_plastic_synapses) {
	int idx = d_plastic_synapse_indices[indx];
        // First decay the memory trace (USING INDX FOR TRACE HERE AND BELOW)
        float stdp_pre_memory_trace_val = stdp_pre_memory_trace[indx];
        float stdp_post_memory_trace_val = stdp_post_memory_trace[indx];
        int postid = d_postsyns[idx];

	for (int g=0; g < timestep_grouping; g++){	
	  stdp_post_memory_trace_val *= expf( - timestep / stdp_vars.tau_minus);
	  stdp_pre_memory_trace_val *= expf(- timestep / stdp_vars.tau_plus);

          // First update the memory trace for every pre and post neuron
          if (fabs((d_time_of_last_spike_to_reach_synapse[idx] + g*timestep) - current_time_in_seconds) < 0.5f*timestep){
            // Update the presynaptic memory trace
            stdp_pre_memory_trace_val += stdp_vars.a_plus;
            // Carry out the necessary LTD
	    float old_synaptic_weight = d_synaptic_efficacies_or_weights[idx];
            d_synaptic_efficacies_or_weights[idx] -= old_synaptic_weight * stdp_post_memory_trace_val;
          }	
          // Dealing with LTP
	  if (fabs((d_last_spike_time_of_each_neuron[postid] + g*timestep) - current_time_in_seconds) < 0.5f*timestep){
            stdp_post_memory_trace_val += stdp_vars.a_minus;
            // If output neuron just fired, do LTP
            d_synaptic_efficacies_or_weights[idx] += stdp_pre_memory_trace_val;
          }
	}

	// Correctly set the trace values
	stdp_pre_memory_trace[indx] = stdp_pre_memory_trace_val;
	stdp_post_memory_trace[indx] = stdp_post_memory_trace_val;

        indx += blockDim.x * gridDim.x;
      }

    }
    
  }
}
