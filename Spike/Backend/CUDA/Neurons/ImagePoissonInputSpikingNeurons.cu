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

    void ImagePoissonInputSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      ::Backend::CUDA::SpikingSynapses* synapses_backend =
        dynamic_cast<::Backend::CUDA::SpikingSynapses*>(frontend()->model->spiking_synapses->backend());
      poisson_update_membrane_potentials_kernel<<<random_state_manager_backend->block_dimensions, random_state_manager_backend->threads_per_block>>>(
         synapses_backend->host_syn_activation_kernel,
         synapses_backend->d_synaptic_data,
         d_neuron_data,
         random_state_manager_backend->states,
         gabor_input_rates,
         active,
         membrane_potentials_v,
         timestep,
         frontend()->model->timestep_grouping,
         thresholds_for_action_potential_spikes,
         resting_potentials_v0,
         last_spike_time_of_each_neuron,
         current_time_in_seconds,
         (int)roundf(current_time_in_seconds / timestep),
         frontend()->total_number_of_neurons,
         frontend()->current_stimulus_index);

      CudaCheckError();
    }
  }
}
