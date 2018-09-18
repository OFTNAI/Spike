// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, GeneratorInputSpikingNeurons)

namespace Backend {
  namespace CUDA {
    namespace INLINE_GENE {
      #include "Spike/Backend/CUDA/InlineDeviceFunctions.hpp"
    }
    GeneratorInputSpikingNeurons::~GeneratorInputSpikingNeurons() {
      CudaSafeCall(cudaFree(neuron_ids_for_stimulus));
      CudaSafeCall(cudaFree(spike_times_for_stimulus));
    }
    
    // Allocate device pointers for the longest stimulus so that they do not need to be replaced
    void GeneratorInputSpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&neuron_ids_for_stimulus, sizeof(int)*frontend()->length_of_longest_stimulus));
      CudaSafeCall(cudaMalloc((void **)&spike_times_for_stimulus, sizeof(float)*frontend()->length_of_longest_stimulus));
    }

    void GeneratorInputSpikingNeurons::prepare() {
      InputSpikingNeurons::prepare();
      allocate_device_pointers();
    }

    void GeneratorInputSpikingNeurons::reset_state() {
      InputSpikingNeurons::reset_state();

      CudaSafeCall(cudaMemcpy(neuron_ids_for_stimulus,
                              frontend()->neuron_id_matrix_for_stimuli[frontend()->current_stimulus_index],
                              sizeof(int)*frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index],
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(spike_times_for_stimulus,
                              frontend()->spike_times_matrix_for_stimuli[frontend()->current_stimulus_index],
                              sizeof(float)*frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index],
                              cudaMemcpyHostToDevice));
      num_spikes_in_current_stimulus = frontend()->number_of_spikes_in_stimuli[frontend()->current_stimulus_index];
    }

    void GeneratorInputSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      if ((frontend()->temporal_lengths_of_stimuli[frontend()->current_stimulus_index] +  timestep) > (current_time_in_seconds - frontend()->stimulus_onset_adjustment)){
        ::Backend::CUDA::SpikingSynapses* synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>(frontend()->model->spiking_synapses->backend());
        check_for_generator_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(
           synapses_backend->host_syn_activation_kernel,
           synapses_backend->d_synaptic_data,
           d_neuron_data,
           neuron_ids_for_stimulus,
           spike_times_for_stimulus,
           last_spike_time_of_each_neuron,
           current_time_in_seconds,
           frontend()->stimulus_onset_adjustment,
           timestep,
           frontend()->model->timestep_grouping,
           num_spikes_in_current_stimulus);

        CudaCheckError();
      }
    }


    __global__ void check_for_generator_spikes_kernel(
        synaptic_activation_kernel syn_activation_kernel,
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        int *d_neuron_ids_for_stimulus,
        float *d_spike_times_for_stimulus,
        float* d_last_spike_time_of_each_neuron,
        float current_time_in_seconds,
        float stimulus_onset_adjustment,
        float timestep,
        int timestep_grouping,
        size_t number_of_spikes_in_stimulus){

      // // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      int bufsize = neuron_data->neuron_spike_time_bitbuffer_bytesize[0];
      while (idx < number_of_spikes_in_stimulus) {
        for (int g=0; g < timestep_grouping; g++){
          int bitloc = ((int)roundf(current_time_in_seconds / timestep) + g) % (8*bufsize);
          neuron_data->neuron_spike_time_bitbuffer[idx*bufsize + (bitloc / 8)] &= ~(1 << (bitloc % 8));
          if (fabs((current_time_in_seconds - stimulus_onset_adjustment + g*timestep) - d_spike_times_for_stimulus[idx]) < 0.5 * timestep) {
            neuron_data->neuron_spike_time_bitbuffer[d_neuron_ids_for_stimulus[idx]*bufsize + (bitloc / 8)] |= (1 << (bitloc % 8));
            #ifndef INLINEDEVICEFUNCS
              syn_activation_kernel(
            #else
              INLINE_GENE::my_activate_synapses(
            #endif
                synaptic_data,
                neuron_data,
                g,
                idx,
                true);
          }
        }
        idx += blockDim.x * gridDim.x;
      }
    }
  }
}
