// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/Neurons.hpp"

namespace Backend {
  namespace CUDA {
    Neurons::~Neurons() {
      CudaSafeCall(cudaFree(per_neuron_afferent_synapse_count));
      CudaSafeCall(cudaFree(current_injections));
    }

    void Neurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps,  bool high_fidelity_spike_storage) {

      CudaSafeCall(cudaMalloc((void **)&current_injections, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&per_neuron_afferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons));
    }

    void Neurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(per_neuron_afferent_synapse_count, frontend()->per_neuron_afferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void Neurons::set_threads_per_block_and_blocks_per_grid(int threads) {
      threads_per_block.x = threads;

      int number_of_neuron_blocks = (frontend()->total_number_of_neurons + threads) / threads;
      number_of_neuron_blocks_per_grid.x = number_of_neuron_blocks;
    }

      /**  
       *  A local, non-polymorphic function called by Neurons::reset_neuron_activities to reset Neurons::d_current_injections.
       */
    void Neurons::reset_current_injections() {
      CudaSafeCall(cudaMemset(current_injections, 0.0f, frontend()->total_number_of_neurons*sizeof(float)));
    }

    void Neurons::reset_state() {
      reset_current_injections();
    }  
  } // ::Backend::CUDA
} // ::Backend
