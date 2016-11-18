#include "Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    ImagePoissonInputSpikingNeurons::~ImagePoissonInputSpikingNeurons() {
      CudaSafeCall(cudaFree(gabor_input_rates));
    }

    void ImagePoissonInputSpikingNeurons::copy_rates_to_device() {
      CudaSafeCall(cudaMalloc((void **)&d_gabor_input_rates, sizeof(float)*total_number_of_rates));
      CudaSafeCall(cudaMemcpy(d_gabor_input_rates, gabor_input_rates, sizeof(float)*total_number_of_rates, cudaMemcpyHostToDevice));
    }

    void ImagePoissonInputSpikingNeurons::update_membrane_potentials(float timestep,float current_time_in_seconds) {
      poisson_update_membrane_potentials_kernel<<<random_state_manager->block_dimensions, random_state_manager->threads_per_block>>>
        (random_state_manager->d_states,
         d_gabor_input_rates,
         d_membrane_potentials_v,
         timestep,
         d_thresholds_for_action_potential_spikes,
         total_number_of_neurons,
         current_stimulus_index);

	CudaCheckError();
    }
  }
}
