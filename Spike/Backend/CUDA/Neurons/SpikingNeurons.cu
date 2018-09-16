// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingNeurons);

namespace Backend {
  namespace CUDA {
    SpikingNeurons::SpikingNeurons() {
    }
    SpikingNeurons::~SpikingNeurons() {
      CudaSafeCall(cudaFree(last_spike_time_of_each_neuron));
      CudaSafeCall(cudaFree(membrane_potentials_v));
      CudaSafeCall(cudaFree(thresholds_for_action_potential_spikes));
      CudaSafeCall(cudaFree(resting_potentials_v0));
      CudaSafeCall(cudaFree(after_spike_reset_potentials_vreset));
      CudaSafeCall(cudaFree(d_neuron_data));

      CudaSafeCall(cudaFree(neuron_spike_time_bitbuffer));
      CudaSafeCall(cudaFree(neuron_spike_time_bitbuffer_currentloc));
      CudaSafeCall(cudaFree(neuron_spike_time_bitbuffer));
    }

    void SpikingNeurons::allocate_device_pointers() {
      
      CudaSafeCall(cudaMalloc((void **)&last_spike_time_of_each_neuron, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&membrane_potentials_v, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&thresholds_for_action_potential_spikes, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&resting_potentials_v0, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&after_spike_reset_potentials_vreset, sizeof(float)*frontend()->total_number_of_neurons));

      CudaSafeCall(cudaMalloc((void **)&d_neuron_data, sizeof(spiking_neurons_data_struct)));
      
      h_neuron_spike_time_bitbuffer_bytesize = (int)ceil((float)(frontend()->model->spiking_synapses->maximum_axonal_delay_in_timesteps + frontend()->model->timestep_grouping) / 8.0f);
      CudaSafeCall(cudaMalloc((void **)&neuron_spike_time_bitbuffer_bytesize, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&neuron_spike_time_bitbuffer_currentloc, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&neuron_spike_time_bitbuffer, sizeof(uint8_t)*h_neuron_spike_time_bitbuffer_bytesize));
    }

    void SpikingNeurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(thresholds_for_action_potential_spikes, frontend()->thresholds_for_action_potential_spikes, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(resting_potentials_v0, frontend()->resting_potentials_v0, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(after_spike_reset_potentials_vreset, frontend()->after_spike_reset_potentials_vreset, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void SpikingNeurons::prepare() {
      Neurons::prepare();
      allocate_device_pointers();
      copy_constants_to_device();

      neuron_data = new spiking_neurons_data_struct();
      memcpy(neuron_data, (static_cast<SpikingNeurons*>(this)->Neurons::neuron_data), sizeof(neurons_data_struct));
      neuron_data->last_spike_time_of_each_neuron = last_spike_time_of_each_neuron;
      neuron_data->membrane_potentials_v = membrane_potentials_v;
      neuron_data->thresholds_for_action_potential_spikes = thresholds_for_action_potential_spikes;
      neuron_data->resting_potentials_v0 = resting_potentials_v0;
      neuron_data->total_number_of_neurons = frontend()->total_number_of_neurons;
      neuron_data->after_spike_reset_potentials_vreset = after_spike_reset_potentials_vreset;

      neuron_data->neuron_spike_time_bitbuffer = neuron_spike_time_bitbuffer;
      neuron_data->neuron_spike_time_bitbuffer_currentloc = neuron_spike_time_bitbuffer_currentloc;
      neuron_data->neuron_spike_time_bitbuffer_bytesize = neuron_spike_time_bitbuffer_bytesize;


      CudaSafeCall(cudaMemcpy(
		d_neuron_data, 
		neuron_data,
		sizeof(spiking_neurons_data_struct), cudaMemcpyHostToDevice));
    }

    void SpikingNeurons::reset_state() {
      Neurons::reset_state();

      // Set last spike times to -1000 so that the times do not affect current simulation.
      float* tmp_last_spike_times;
      tmp_last_spike_times = (float*)malloc(sizeof(float)*frontend()->total_number_of_neurons);
      for (int i=0; i < frontend()->total_number_of_neurons; i++){
        tmp_last_spike_times[i] = -1000.0f;
      }

      CudaSafeCall(cudaMemcpy(last_spike_time_of_each_neuron,
                              tmp_last_spike_times,
                              frontend()->total_number_of_neurons*sizeof(float),
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(membrane_potentials_v,
                              frontend()->resting_potentials_v0,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));

      CudaSafeCall(cudaMemset(neuron_spike_time_bitbuffer_currentloc, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(neuron_spike_time_bitbuffer, 0, sizeof(uint8_t)*h_neuron_spike_time_bitbuffer_bytesize));
      CudaSafeCall(cudaMemcpy(neuron_spike_time_bitbuffer_bytesize,
                              &h_neuron_spike_time_bitbuffer_bytesize,
                              sizeof(int),
                              cudaMemcpyHostToDevice));
      // Free tmp_last_spike_times
      free (tmp_last_spike_times);
    }
    
    void SpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
    }


  } // ::Backend::CUDA
} // ::Backend
