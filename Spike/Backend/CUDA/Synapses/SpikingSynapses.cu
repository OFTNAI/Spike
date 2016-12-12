// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"
#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/ImagePoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    SpikingSynapses::~SpikingSynapses() {
      CudaSafeCall(cudaFree(delays));
      CudaSafeCall(cudaFree(spikes_travelling_to_synapse));
      CudaSafeCall(cudaFree(stdp));
      CudaSafeCall(cudaFree(time_of_last_spike_to_reach_synapse));
    }

    void SpikingSynapses::reset_state() {
      CudaSafeCall(cudaMemset(spikes_travelling_to_synapse, 0,
                              sizeof(int)*frontend()->total_number_of_synapses));

      // Set last spike times to -1000 so that the times do not affect current simulation.
      float last_spike_to_reach_synapse[frontend()->total_number_of_synapses];
      for (int i=0; i < frontend()->total_number_of_synapses; i++)
        last_spike_to_reach_synapse[i] = -1000.0f;

      CudaSafeCall(cudaMemcpy(time_of_last_spike_to_reach_synapse,
                              last_spike_to_reach_synapse,
                              frontend()->total_number_of_synapses*sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void SpikingSynapses::push_data_front() {
      // TODO: Flesh this out (and for derived classes!)
      CudaSafeCall(cudaMemcpy(frontend()->synaptic_efficacies_or_weights,
                              synaptic_efficacies_or_weights,
                              sizeof(float)*frontend()->total_number_of_synapses,
                              cudaMemcpyDeviceToHost));
    }

    void SpikingSynapses::allocate_device_pointers() {
      Synapses::allocate_device_pointers();

      CudaSafeCall(cudaMalloc((void **)&delays, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&stdp, sizeof(bool)*frontend()->total_number_of_synapses));

      CudaSafeCall(cudaMalloc((void **)&spikes_travelling_to_synapse, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&time_of_last_spike_to_reach_synapse, sizeof(float)*frontend()->total_number_of_synapses));
    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      Synapses::copy_constants_and_initial_efficacies_to_device();

      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(stdp, frontend()->stdp,
                              sizeof(bool)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
    }

    void SpikingSynapses::interact_spikes_with_synapses
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {

      // TODO: Fix this weird cludge (want to cast most generically!)
      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::LIFSpikingNeurons*>(neurons->backend());
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::ImagePoissonInputSpikingNeurons*>(input_neurons->backend());

      std::cout << "########## " << TYPEID_NAME(input_neurons->backend()) << "\n"
                << input_neurons_backend << " (" << input_neurons->_backend << ") -- "
                << input_neurons_backend->membrane_potentials_v << "\n";

      printf(";;;;;;;; %p, %p, %p, %p, %d, %d, %f, %f, %d, %p\n",
             presynaptic_neuron_indices,
             delays,
             neurons_backend->bitarray_of_neuron_spikes,
             input_neurons_backend->bitarray_of_neuron_spikes,
             neurons->bitarray_length,
             neurons->bitarray_maximum_axonal_delay_in_timesteps,
             current_time_in_seconds,
             timestep,
             frontend()->total_number_of_synapses,
             time_of_last_spike_to_reach_synapse);
      
      if (neurons->high_fidelity_spike_flag){
        check_bitarray_for_presynaptic_neuron_spikes<<<number_of_synapse_blocks_per_grid, threads_per_block>>>
          (presynaptic_neuron_indices,
           delays,
           neurons_backend->bitarray_of_neuron_spikes,
           input_neurons_backend->bitarray_of_neuron_spikes,
           neurons->bitarray_length,
           neurons->bitarray_maximum_axonal_delay_in_timesteps,
           current_time_in_seconds,
           timestep,
           frontend()->total_number_of_synapses,
           time_of_last_spike_to_reach_synapse);
        CudaCheckError();
      } else {
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

        d_spikes_travelling_to_synapse[idx] = timesteps_until_spike_reaches_synapse;

        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }

    __global__ void check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
                                                                 int* d_delays,
                                                                 unsigned char* d_bitarray_of_neuron_spikes,
                                                                 unsigned char* d_input_neuron_bitarray_of_neuron_spikes,
                                                                 int bitarray_length,
                                                                 int bitarray_maximum_axonal_delay_in_timesteps,
                                                                 float current_time_in_seconds,
                                                                 float timestep,
                                                                 size_t total_number_of_synapses,
                                                                 float* d_time_of_last_spike_to_reach_synapse){
	
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_synapses) {

        int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
        int delay = d_delays[idx];

        // Get offset depending upon the current timestep
        int offset_index = ((int)(round(current_time_in_seconds / timestep)) % bitarray_maximum_axonal_delay_in_timesteps) - delay;
        offset_index = (offset_index < 0) ? (offset_index + bitarray_maximum_axonal_delay_in_timesteps) : offset_index;
        int offset_byte = offset_index / 8;
        int offset_bit_pos = offset_index - (8 * offset_byte);

        // Get the correct neuron index
        int neuron_index = CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input);
		
        // Check the spike
        int neuron_id_spike_store_start = neuron_index * bitarray_length;
        int check = 0;
        if (presynaptic_is_input){
          unsigned char byte = d_input_neuron_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          check = ((byte >> offset_bit_pos) & 1);
          if (check == 1){
            d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
          }
        } else {
          unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          check = ((byte >> offset_bit_pos) & 1);
          if (check == 1){
            d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
          }
        }

        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }
  }
}
