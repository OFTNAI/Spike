// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/ImagePoissonInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, ImagePoissonInputSpikingNeurons);

namespace Backend {
  namespace CUDA {
    ImagePoissonInputSpikingNeurons::~ImagePoissonInputSpikingNeurons() {
      CudaSafeCall(cudaFree(gabor_input_rates));
    }

    void ImagePoissonInputSpikingNeurons::allocate_device_pointers() {
    }

    void ImagePoissonInputSpikingNeurons::copy_rates_to_device() {
      if (gabor_input_rates)
        CudaSafeCall(cudaFree(gabor_input_rates));
      CudaSafeCall(cudaMalloc((void **)&gabor_input_rates, sizeof(float)*frontend()->total_number_of_rates));
      CudaSafeCall(cudaMemcpy(gabor_input_rates, frontend()->gabor_input_rates, sizeof(float)*frontend()->total_number_of_rates, cudaMemcpyHostToDevice));
    }

    void ImagePoissonInputSpikingNeurons::reset_state() {
      PoissonInputSpikingNeurons::reset_state();
    }

    void ImagePoissonInputSpikingNeurons::prepare() {
      PoissonInputSpikingNeurons::prepare();
      allocate_device_pointers();
      copy_rates_to_device();
    }

    void ImagePoissonInputSpikingNeurons::push_data_front() {
      PoissonInputSpikingNeurons::push_data_front();
    }

    void ImagePoissonInputSpikingNeurons::pull_data_back() {
      PoissonInputSpikingNeurons::pull_data_back();
    }

    void ImagePoissonInputSpikingNeurons::update_membrane_potentials(float timestep,float current_time_in_seconds) {
      poisson_update_membrane_potentials_kernel<<<random_state_manager_backend->block_dimensions, random_state_manager_backend->threads_per_block>>>
        (random_state_manager_backend->states,
         gabor_input_rates,
         membrane_potentials_v,
         timestep,
         thresholds_for_action_potential_spikes,
         frontend()->total_number_of_neurons,
         frontend()->current_stimulus_index);

	CudaCheckError();
    }
  }
}
