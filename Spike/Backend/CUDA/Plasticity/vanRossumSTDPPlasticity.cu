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
      
      if (frontend()->stdp_params->allspikes){
        CudaSafeCall(cudaMemcpy((void*)stdp_pre_memory_trace,
                                (void*)h_stdp_trace,
                                sizeof(float)*total_number_of_plastic_synapses,
                                cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy((void*)stdp_post_memory_trace,
                                (void*)h_stdp_trace,
                                sizeof(float)*total_number_of_plastic_synapses,
                                cudaMemcpyHostToDevice));
      } else {
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
    }

    void vanRossumSTDPPlasticity::prepare() {
      STDPPlasticity::prepare();

      allocate_device_pointers();
    }

    void vanRossumSTDPPlasticity::allocate_device_pointers() {
      // The following doesn't do anything in original code...
      // ::Backend::CUDA::STDPPlasticity::allocate_device_pointers();
      if (frontend()->stdp_params->allspikes){
        CudaSafeCall(cudaMalloc((void **)&stdp_pre_memory_trace, sizeof(float)*total_number_of_plastic_synapses));
        CudaSafeCall(cudaMalloc((void **)&stdp_post_memory_trace, sizeof(float)*total_number_of_plastic_synapses));
        // Also allocate and set trace template
	h_stdp_trace = (float *)malloc( sizeof(float) * total_number_of_plastic_synapses);
        for (int i=0; i < total_number_of_plastic_synapses; i++){
          h_stdp_trace[i] = 0.0f;
        }
      } else {
        CudaSafeCall(cudaMalloc((void **)&index_of_last_afferent_synapse_to_spike, sizeof(int)*frontend()->neurs->total_number_of_neurons));
        CudaSafeCall(cudaMalloc((void **)&isindexed_ltd_synapse_spike, sizeof(int)*frontend()->neurs->total_number_of_neurons));
        CudaSafeCall(cudaMalloc((void **)&index_of_first_synapse_spiked_after_postneuron, sizeof(int)*frontend()->neurs->total_number_of_neurons));
      }
    }

    void vanRossumSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds) {
      if (total_number_of_plastic_synapses > 0){ 
      // If nearest spike, get the indices:
      if (!frontend()->stdp_params->allspikes){ 
        vanrossum_get_indices_to_apply_stdp<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
          (synapses_backend->postsynaptic_neuron_indices,
           neurons_backend->last_spike_time_of_each_neuron,
           synapses_backend->time_of_last_spike_to_reach_synapse,
           index_of_last_afferent_synapse_to_spike,
           isindexed_ltd_synapse_spike,
           index_of_first_synapse_spiked_after_postneuron,
           current_time_in_seconds,
           plastic_synapse_indices,
           total_number_of_plastic_synapses);
        CudaCheckError();

        // Either way, run the synapse update
        vanrossum_apply_stdp_to_synapse_weights_kernel_nearest<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
          (synapses_backend->postsynaptic_neuron_indices,
           neurons_backend->last_spike_time_of_each_neuron,
           synapses_backend->time_of_last_spike_to_reach_synapse,
           synapses_backend->synaptic_efficacies_or_weights,
           index_of_last_afferent_synapse_to_spike,
           isindexed_ltd_synapse_spike,
           index_of_first_synapse_spiked_after_postneuron,
           *(frontend()->stdp_params),
           current_time_in_seconds,
           frontend()->neurs->total_number_of_neurons);
        CudaCheckError();
      } else {
        vanrossum_ltp_and_ltd<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
          (synapses_backend->postsynaptic_neuron_indices,
           synapses_backend->time_of_last_spike_to_reach_synapse,
           neurons_backend->last_spike_time_of_each_neuron,
           synapses_backend->synaptic_efficacies_or_weights,
           stdp_pre_memory_trace,
           stdp_post_memory_trace,
           *(frontend()->stdp_params),
           frontend()->stdp_params->timestep,
           current_time_in_seconds,
           plastic_synapse_indices,
           total_number_of_plastic_synapses);
        
      }
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
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses){
      // Global Index
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all neurons
      while (indx < total_number_of_plastic_synapses) {
	int idx = d_plastic_synapse_indices[indx];
        // First decay the memory trace (USING INDX FOR TRACE HERE AND BELOW)
        stdp_pre_memory_trace[indx] *= expf(- timestep / stdp_vars.tau_plus);
        stdp_post_memory_trace[indx] *= expf( - timestep / stdp_vars.tau_minus);

        int postid = d_postsyns[idx];
        // First update the memory trace for every pre and post neuron
        if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds){
          // Update the presynaptic memory trace
          stdp_pre_memory_trace[indx] += stdp_vars.a_plus;
          // Carry out the necessary LTD
          int postid = d_postsyns[idx];
	  float old_synaptic_weight = d_synaptic_efficacies_or_weights[idx];
          d_synaptic_efficacies_or_weights[idx] -= old_synaptic_weight * stdp_post_memory_trace[postid];
        }	
        // Dealing with LTP
	if (d_last_spike_time_of_each_neuron[postid] == current_time_in_seconds){
          stdp_post_memory_trace[indx] += stdp_vars.a_minus;
          // If output neuron just fired, do LTP
          d_synaptic_efficacies_or_weights[idx] += stdp_pre_memory_trace[indx];
        }
        indx += blockDim.x * gridDim.x;
      }

    }
    
    // NEAREST SPIKE IMPLEMENTATION
    __global__ void vanrossum_apply_stdp_to_synapse_weights_kernel_nearest
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     float* d_time_of_last_spike_to_reach_synapse,
     float* d_synaptic_efficacies_or_weights,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     struct vanrossum_stdp_plasticity_parameters_struct stdp_vars,
     float currtime,
     size_t total_number_of_post_neurons){
      // Global Index
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all neurons
      while (idx < total_number_of_post_neurons) {
        // Check whether a neuron has fired, if so: reset flag
        if (d_last_spike_time_of_each_neuron[idx] == currtime){
          d_isindexed_ltd_synapse_spike[idx] = false;
        }

        // Get the synapse on which to do LTP/LTD
        int index_of_LTP_synapse = d_index_of_last_afferent_synapse_to_spike[idx];
        int index_of_LTD_synapse = d_index_of_first_synapse_spiked_after_postneuron[idx];

        // If we are to carry out STDP on LTP synapse
        if (index_of_LTP_synapse >= 0){
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
            float last_syn_spike_time = d_time_of_last_spike_to_reach_synapse[index_of_LTD_synapse];
            float last_neuron_spike_time = d_last_spike_time_of_each_neuron[idx];
            float new_syn_weight = d_synaptic_efficacies_or_weights[index_of_LTD_synapse];
 
            // Set the index to negative (i.e. Reset it)
            d_index_of_first_synapse_spiked_after_postneuron[idx] = -1;

            float diff = last_syn_spike_time - last_neuron_spike_time;
            // Only carry out LTD if the difference is in some range
            if (diff < 7*stdp_vars.tau_minus && diff > 0){
              float weightchange = powf(new_syn_weight, stdp_vars.weight_dependency_factor) * stdp_vars.a_minus * expf(-diff / stdp_vars.tau_minus);
              // Update the weights
              new_syn_weight -= weightchange;
            }
            d_synaptic_efficacies_or_weights[index_of_LTD_synapse] = new_syn_weight;
            
          }
        }
        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }

    __global__ void vanrossum_get_indices_to_apply_stdp
    (int* d_postsyns,
     float* d_last_spike_time_of_each_neuron,
     float* d_time_of_last_spike_to_reach_synapse,
     int* d_index_of_last_afferent_synapse_to_spike,
     bool* d_isindexed_ltd_synapse_spike,
     int* d_index_of_first_synapse_spiked_after_postneuron,
     float currtime,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses){
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running through all synapses:
      while (indx < total_number_of_plastic_synapses){
        int idx = d_plastic_synapse_indices[indx];
        int postsynaptic_neuron = d_postsyns[idx];

        // Check whether a synapse reached a neuron this timestep
        if (d_time_of_last_spike_to_reach_synapse[idx] == currtime){
          // Atomic Exchange the new synapse index
          atomicExch(&d_index_of_last_afferent_synapse_to_spike[postsynaptic_neuron], idx);
        }

        // Check (if we need to) whether a synapse has fired
        if (!d_isindexed_ltd_synapse_spike[postsynaptic_neuron]){
          if (d_time_of_last_spike_to_reach_synapse[idx] == currtime){
            d_isindexed_ltd_synapse_spike[postsynaptic_neuron] = true;
            atomicExch(&d_index_of_first_synapse_spiked_after_postneuron[postsynaptic_neuron], idx);
          }
        }
        // Increment index
        indx += blockDim.x * gridDim.x;
      }
    }
  }
}
