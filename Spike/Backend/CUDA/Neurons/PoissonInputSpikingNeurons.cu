// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, PoissonInputSpikingNeurons);

namespace Backend {
  namespace CUDA {
    namespace INLINE_POIS {
      #include "Spike/Backend/CUDA/InlineDeviceFunctions.hpp"
    }
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
      ::Backend::CUDA::SpikingSynapses* synapses_backend =
        dynamic_cast<::Backend::CUDA::SpikingSynapses*>(frontend()->model->spiking_synapses->backend());
      poisson_update_membrane_potentials_kernel<<<random_state_manager_backend->block_dimensions, random_state_manager_backend->threads_per_block>>>(
         synapses_backend->host_syn_activation_kernel,
         synapses_backend->d_synaptic_data,
         d_neuron_data,
         random_state_manager_backend->states,
         rates,
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

    __global__ void poisson_update_membrane_potentials_kernel(
        synaptic_activation_kernel syn_activation_kernel,
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* in_neuron_data,
        curandState_t* d_states,
       float *d_rates,
       float *d_membrane_potentials_v,
       float timestep,
       int timestep_grouping,
       float * d_thresholds_for_action_potential_spikes,
       float* d_resting_potentials,
       float* d_last_spike_time_of_each_neuron,
       float current_time_in_seconds,
       int timestep_index,
       size_t total_number_of_input_neurons,
       int current_stimulus_index) {

   
      int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
      int idx = t_idx;
      int bufsize = in_neuron_data->neuron_spike_time_bitbuffer_bytesize[0];

      while (idx < total_number_of_input_neurons){

        int rate_index = (total_number_of_input_neurons * current_stimulus_index) + idx;
        float rate = d_rates[rate_index];

        if (rate > 0.01) {
          for (int g=0; g < timestep_grouping; g++){
            int bitloc = (timestep_index + g) % (8*bufsize);
            in_neuron_data->neuron_spike_time_bitbuffer[idx*bufsize + (bitloc / 8)] &= ~(1 << (bitloc % 8));
            // Creates random float between 0 and 1 from uniform distribution
            // d_states effectively provides a different seed for each thread
            // curand_uniform produces different float every time you call it
            float random_float = curand_uniform(&d_states[t_idx]);
      
            // if the randomnumber is less than the rate
            if (random_float < (rate * timestep)) {
              in_neuron_data->neuron_spike_time_bitbuffer[idx*bufsize + (bitloc / 8)] |= (1 << (bitloc % 8));
              #ifndef INLINEDEVICEFUNCS
                syn_activation_kernel(
              #else
                INLINE_POIS::my_activate_synapses(
              #endif
                  synaptic_data,
                  in_neuron_data,
                  g,
                  idx,
                  timestep_index / timestep_grouping,
                  true);
            } 
          }
        }

        idx += blockDim.x * gridDim.x;

      }
    }
  }
}
