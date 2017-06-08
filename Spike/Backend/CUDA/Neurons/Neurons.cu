// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/Neurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, Neurons);

namespace Backend {
  namespace CUDA {
    Neurons::~Neurons() {
      CudaSafeCall(cudaFree(per_neuron_afferent_synapse_count));
      CudaSafeCall(cudaFree(current_injections));
      CudaSafeCall(cudaFree(per_neuron_efferent_synapse_count));
      CudaSafeCall(cudaFree(per_neuron_efferent_synapse_total));
      CudaSafeCall(cudaFree(per_neuron_efferent_synapse_indices));
      free(h_per_neuron_efferent_synapse_total);
    }

    void Neurons::allocate_device_pointers() {
      h_per_neuron_efferent_synapse_total = (int*)malloc(sizeof(int)*frontend()->total_number_of_neurons);
      for (int i = 0; i < frontend()->total_number_of_neurons; i++){
        if (i == 0)
          h_per_neuron_efferent_synapse_total[i] = frontend()->per_neuron_efferent_synapse_count[i];
	else
	  h_per_neuron_efferent_synapse_total[i] = h_per_neuron_efferent_synapse_total[i-1] + frontend()->per_neuron_efferent_synapse_count[i];
      }
      CudaSafeCall(cudaMalloc((void **)&current_injections, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&per_neuron_afferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&per_neuron_efferent_synapse_total, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&per_neuron_efferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&per_neuron_efferent_synapse_indices, sizeof(int)*(h_per_neuron_efferent_synapse_total[frontend()->total_number_of_neurons - 1])));
    }

    void Neurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(per_neuron_afferent_synapse_count, frontend()->per_neuron_afferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(per_neuron_efferent_synapse_total, h_per_neuron_efferent_synapse_total, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(per_neuron_efferent_synapse_count, frontend()->per_neuron_efferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      for (int i = 0; i < frontend()->total_number_of_neurons; i++)
	CudaSafeCall(cudaMemcpy(&per_neuron_efferent_synapse_indices[h_per_neuron_efferent_synapse_total[i] - frontend()->per_neuron_efferent_synapse_count[i]], frontend()->per_neuron_efferent_synapse_indices[i], sizeof(int)*frontend()->per_neuron_efferent_synapse_count[i], cudaMemcpyHostToDevice));
	
    }

    void Neurons::set_threads_per_block_and_blocks_per_grid(int threads) {
      threads_per_block.x = threads;
      cudaDeviceProp deviceProp;
      int deviceID;

      cudaGetDevice(&deviceID);
      cudaGetDeviceProperties(&deviceProp, deviceID);

      int max_num_blocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor / threads);

      int number_of_neuron_blocks = (frontend()->total_number_of_neurons + threads) / threads;
      number_of_neuron_blocks_per_grid = dim3(number_of_neuron_blocks);
      if (number_of_neuron_blocks > max_num_blocks)
	number_of_neuron_blocks_per_grid = dim3(max_num_blocks);
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
  } // ::Backend::CUDA
} // ::Backend
