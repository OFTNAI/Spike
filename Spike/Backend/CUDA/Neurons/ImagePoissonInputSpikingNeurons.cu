// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/ImagePoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    ImagePoissonInputSpikingNeurons::~ImagePoissonInputSpikingNeurons() {
      CudaSafeCall(cudaFree(gabor_input_rates));
    }

    void ImagePoissonInputSpikingNeurons::copy_rates_to_device() {
      CudaSafeCall(cudaMalloc((void **)&gabor_input_rates, sizeof(float)*frontend()->total_number_of_rates));
      CudaSafeCall(cudaMemcpy(gabor_input_rates, frontend()->gabor_input_rates, sizeof(float)*frontend()->total_number_of_rates, cudaMemcpyHostToDevice));
    }

    void ImagePoissonInputSpikingNeurons::update_membrane_potentials(float timestep,float current_time_in_seconds) {

      // Crudely assume that the RandomStateManager backend is also CUDA:
      ::Backend::CUDA::RandomStateManager* random_state_manager
        = static_cast<::Backend::CUDA::RandomStateManager*>
        (frontend()->random_state_manager->backend());

      poisson_update_membrane_potentials_kernel<<<random_state_manager->block_dimensions, random_state_manager->threads_per_block>>>
        (random_state_manager->states,
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
