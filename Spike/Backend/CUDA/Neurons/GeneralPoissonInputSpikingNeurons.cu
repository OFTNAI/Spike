// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/GeneralPoissonInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, GeneralPoissonInputSpikingNeurons);

namespace Backend {
  namespace CUDA {
    GeneralPoissonInputSpikingNeurons::~GeneralPoissonInputSpikingNeurons() {
      CudaSafeCall(cudaFree(stimuli_rates));
    }

    void GeneralPoissonInputSpikingNeurons::allocate_device_pointers() {
    }

    void GeneralPoissonInputSpikingNeurons::copy_rates_to_device() {
      if (stimuli_rates)
        CudaSafeCall(cudaFree(stimuli_rates));
      CudaSafeCall(cudaMalloc((void **)&stimuli_rates, sizeof(float)*frontend()->total_number_of_rates));
      CudaSafeCall(cudaMemcpy(stimuli_rates, frontend()->stimuli_rates, sizeof(float)*frontend()->total_number_of_rates, cudaMemcpyHostToDevice));
    }

    void GeneralPoissonInputSpikingNeurons::reset_state() {
      PoissonInputSpikingNeurons::reset_state();
    }

    void GeneralPoissonInputSpikingNeurons::prepare() {
      PoissonInputSpikingNeurons::prepare();
      allocate_device_pointers();
      copy_rates_to_device();
    }

    void GeneralPoissonInputSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      poisson_update_membrane_potentials_kernel<<<random_state_manager_backend->block_dimensions, random_state_manager_backend->threads_per_block>>>
        (random_state_manager_backend->states,
         stimuli_rates,
         membrane_potentials_v,
         timestep,
         thresholds_for_action_potential_spikes,
	 resting_potentials,
	 last_spike_time_of_each_neuron,
	 current_time_in_seconds,
         frontend()->total_number_of_neurons,
         frontend()->current_stimulus_index);

	CudaCheckError();
    }
  }
}
