// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/Neurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, Neurons);

namespace Backend {
  namespace CUDA {
    Neurons::~Neurons() {
      CudaSafeCall(cudaFree(per_neuron_afferent_synapse_count));
      CudaSafeCall(cudaFree(current_injections));
    }

    void Neurons::allocate_device_pointers() {
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

    void Neurons::prepare() {
      set_threads_per_block_and_blocks_per_grid(context->params.threads_per_block_neurons);
      allocate_device_pointers();
      copy_constants_to_device();
    }

    void Neurons::reset_current_injections() {
      CudaSafeCall(cudaMemset(current_injections, 0.0f, frontend()->total_number_of_neurons*sizeof(float)));
    }

    void Neurons::reset_state() {
      reset_current_injections();
    }  

    void Neurons::push_data_front() {
    }

    void Neurons::pull_data_back() {
    }
  } // ::Backend::CUDA
} // ::Backend
