// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingSynapses);

namespace Backend {
  namespace CUDA {
    SpikingSynapses::~SpikingSynapses() {
      CudaSafeCall(cudaFree(delays));
      CudaSafeCall(cudaFree(spikes_travelling_to_synapse));
      CudaSafeCall(cudaFree(time_of_last_spike_to_reach_synapse));
#ifdef CRAZY_DEBUG
      std::cout << "\n!!!!!!!!!!!!!!!!!!!!---BBBBBB---!!!!!!!!!!!!!!!!!!!\n";
#endif
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();

      CudaSafeCall(cudaMemset(spikes_travelling_to_synapse, 0,
                              sizeof(int)*frontend()->total_number_of_synapses));

      // Set last spike times to -1000 so that the times do not affect current simulation.
      float *last_spike_to_reach_synapse = (float*)malloc(frontend()->total_number_of_synapses*sizeof(float));
      for (int i=0; i < frontend()->total_number_of_synapses; i++)
        last_spike_to_reach_synapse[i] = -1000.0f;

      CudaSafeCall(cudaMemcpy(time_of_last_spike_to_reach_synapse,
                              last_spike_to_reach_synapse,
                              frontend()->total_number_of_synapses*sizeof(float),
                              cudaMemcpyHostToDevice));
      free(last_spike_to_reach_synapse);
    }

    void SpikingSynapses::copy_weights_to_host() {
      CudaSafeCall(cudaMemcpy(frontend()->synaptic_efficacies_or_weights,
                              synaptic_efficacies_or_weights,
                              sizeof(float)*frontend()->total_number_of_synapses,
                              cudaMemcpyDeviceToHost));
    }

    void SpikingSynapses::prepare() {
      Synapses::prepare();
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
    }

    void SpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&delays, sizeof(int)*frontend()->total_number_of_synapses));

      CudaSafeCall(cudaMalloc((void **)&spikes_travelling_to_synapse, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&time_of_last_spike_to_reach_synapse, sizeof(float)*frontend()->total_number_of_synapses));
    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
    }

    void SpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {

      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons->backend());
      assert(neurons_backend);
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(input_neurons->backend());
      assert(input_neurons_backend);

        move_spikes_towards_synapses_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>
          (presynaptic_neuron_indices,
           delays,
           spikes_travelling_to_synapse,
           neurons_backend->last_spike_time_of_each_neuron,
           input_neurons_backend->last_spike_time_of_each_neuron,
           current_time_in_seconds,
           frontend()->total_number_of_synapses,
           time_of_last_spike_to_reach_synapse);
        CudaCheckError();
    }

    __global__ void move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
                                                        int* d_delays,
                                                        int* d_spikes_travelling_to_synapse,
                                                        float* d_last_spike_time_of_each_neuron,
                                                        float* d_input_neurons_last_spike_time,
                                                        float current_time_in_seconds,
                                                        size_t total_number_of_synapses,
                                                        float* d_time_of_last_spike_to_reach_synapse){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_synapses) {
        int timesteps_until_spike_reaches_synapse = d_spikes_travelling_to_synapse[idx];
        timesteps_until_spike_reaches_synapse -= 1;

        if (timesteps_until_spike_reaches_synapse == 0) {
          d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
        }

        if (timesteps_until_spike_reaches_synapse < 0) {

          // Get presynaptic neurons last spike time
          int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
          bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
          float presynaptic_neurons_last_spike_time = presynaptic_is_input ? d_input_neurons_last_spike_time[CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input)] : d_last_spike_time_of_each_neuron[presynaptic_neuron_index];

          if (presynaptic_neurons_last_spike_time == current_time_in_seconds){
            timesteps_until_spike_reaches_synapse = d_delays[idx];
          }
        } 

      	__syncthreads();
        d_spikes_travelling_to_synapse[idx] = timesteps_until_spike_reaches_synapse;

        idx += blockDim.x * gridDim.x;
      }
    }

  }
}
