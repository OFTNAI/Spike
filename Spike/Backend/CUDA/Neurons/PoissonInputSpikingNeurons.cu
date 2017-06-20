// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, PoissonInputSpikingNeurons);

namespace Backend {
  namespace CUDA {
    PoissonInputSpikingNeurons::~PoissonInputSpikingNeurons() {
      CudaSafeCall(cudaFree(rates));
    }

    void PoissonInputSpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&rates, sizeof(float)*frontend()->total_number_of_neurons));
    }

    void PoissonInputSpikingNeurons::copy_constants_to_device() {
      if (frontend()->rates) {
        CudaSafeCall(cudaMemcpy(rates, frontend()->rates, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      }
    }

    void PoissonInputSpikingNeurons::reset_state() {
      InputSpikingNeurons::reset_state();
    }

    void PoissonInputSpikingNeurons::prepare() {
      InputSpikingNeurons::prepare();

      allocate_device_pointers();
      copy_constants_to_device();

      // Crudely assume that the RandomStateManager backend is also CUDA:
      random_state_manager_backend
        = dynamic_cast<::Backend::CUDA::RandomStateManager*>
        (frontend()->random_state_manager->backend());
      assert(random_state_manager_backend);
    }

    void PoissonInputSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      poisson_update_membrane_potentials_kernel<<<random_state_manager_backend->block_dimensions, random_state_manager_backend->threads_per_block>>>
        (random_state_manager_backend->states,
         rates,
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

    __global__ void poisson_update_membrane_potentials_kernel(curandState_t* d_states,
                                                              float *d_rates,
                                                              float *d_membrane_potentials_v,
                                                              float timestep,
                                                              float * d_thresholds_for_action_potential_spikes,
							      float* d_resting_potentials,
							      float* d_last_spike_time_of_each_neuron,
							      float current_time_in_seconds,
                                                              size_t total_number_of_input_neurons,
                                                              int current_stimulus_index) {

	 
      int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
      int idx = t_idx;
      while (idx < total_number_of_input_neurons){

        int rate_index = (total_number_of_input_neurons * current_stimulus_index) + idx;

        float rate = d_rates[rate_index];

        if (rate > 0.1) {
          // Creates random float between 0 and 1 from uniform distribution
          // d_states effectively provides a different seed for each thread
          // curand_uniform produces different float every time you call it
          float random_float = curand_uniform(&d_states[t_idx]);
			
          // if the randomnumber is less than the rate
          if (random_float < (rate * timestep)) {
            // Puts membrane potential above default spiking threshold
		d_last_spike_time_of_each_neuron[idx] = current_time_in_seconds;
          } 
        }

        idx += blockDim.x * gridDim.x;

      }
      __syncthreads();
    }
  }
}
