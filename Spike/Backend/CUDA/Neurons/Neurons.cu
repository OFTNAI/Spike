// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/Neurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, Neurons);

namespace Backend {
  namespace CUDA {
    Neurons::~Neurons() {
      CudaSafeCall(cudaFree(per_neuron_afferent_synapse_count));
      CudaSafeCall(cudaFree(per_neuron_efferent_synapse_count));
      CudaSafeCall(cudaFree(per_neuron_efferent_synapse_start));
      CudaSafeCall(cudaFree(d_neuron_data));
    }

    void Neurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&per_neuron_afferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&per_neuron_efferent_synapse_start, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&per_neuron_efferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_neuron_data, sizeof(neurons_data_struct)));
    }

    void Neurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(per_neuron_afferent_synapse_count, frontend()->per_neuron_afferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(per_neuron_efferent_synapse_start, frontend()->per_neuron_efferent_synapse_start, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(per_neuron_efferent_synapse_count, frontend()->per_neuron_efferent_synapse_count, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void Neurons::set_threads_per_block_and_blocks_per_grid(int threads) {
      threads_per_block.x = threads;
      cudaDeviceProp deviceProp;
      int deviceID;

      cudaGetDevice(&deviceID);
      cudaGetDeviceProperties(&deviceProp, deviceID);

      int max_num_blocks = (deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor / threads));

      int number_of_neuron_blocks = (frontend()->total_number_of_neurons + threads) / threads;
      number_of_neuron_blocks_per_grid = dim3(number_of_neuron_blocks);
      if (number_of_neuron_blocks > max_num_blocks)
	number_of_neuron_blocks_per_grid = dim3(max_num_blocks);
    }

    void Neurons::prepare() {
      set_threads_per_block_and_blocks_per_grid(context->params.threads_per_block_neurons);
      allocate_device_pointers();
      copy_constants_to_device();

      neuron_data = new neurons_data_struct();
      neuron_data->total_number_of_neurons = frontend()->total_number_of_neurons;
      neuron_data->per_neuron_efferent_synapse_count = per_neuron_efferent_synapse_count;
      neuron_data->per_neuron_efferent_synapse_start = per_neuron_efferent_synapse_start;
      CudaSafeCall(cudaMemcpy(
		d_neuron_data,
		neuron_data,
		sizeof(neurons_data_struct), cudaMemcpyHostToDevice));

    }

    void Neurons::reset_state() {
      CudaSafeCall(cudaMemcpy(per_neuron_efferent_synapse_start, frontend()->per_neuron_efferent_synapse_start, sizeof(int)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }  
  } // ::Backend::CUDA
} // ::Backend
