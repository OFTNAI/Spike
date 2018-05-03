// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingNeurons);

namespace Backend {
  namespace CUDA {
    SpikingNeurons::~SpikingNeurons() {
      CudaSafeCall(cudaFree(current_injections));
      CudaSafeCall(cudaFree(total_current_conductance));
      CudaSafeCall(cudaFree(last_spike_time_of_each_neuron));
      CudaSafeCall(cudaFree(membrane_potentials_v));
      CudaSafeCall(cudaFree(thresholds_for_action_potential_spikes));
      CudaSafeCall(cudaFree(resting_potentials));
    }

    void SpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&last_spike_time_of_each_neuron, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&membrane_potentials_v, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&thresholds_for_action_potential_spikes, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&resting_potentials, sizeof(float)*frontend()->total_number_of_neurons));
     
      // Creating timestep grouping based data for current injection values 
      CudaSafeCall(cudaMalloc((void **)&current_injections, sizeof(float)*(frontend()->total_number_of_neurons*frontend()->model->timestep_grouping)));
      CudaSafeCall(cudaMalloc((void **)&total_current_conductance, sizeof(float)*(frontend()->total_number_of_neurons*frontend()->model->timestep_grouping)));
    }

    void SpikingNeurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(thresholds_for_action_potential_spikes, frontend()->thresholds_for_action_potential_spikes, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(resting_potentials, frontend()->after_spike_reset_membrane_potentials_c, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void SpikingNeurons::prepare() {
      Neurons::prepare();
      allocate_device_pointers();
      copy_constants_to_device();

      neuron_data = new spiking_neurons_data_struct();
      (neurons_data_struct)*neuron_data = *(static_cast<SpikingNeurons*>(this)->Neurons::neuron_data);
      neuron_data->last_spike_time_of_each_neuron = last_spike_time_of_each_neuron;
      neuron_data->membrane_potentials_v = membrane_potentials_v;
      neuron_data->thresholds_for_action_potential_spikes = thresholds_for_action_potential_spikes;
      neuron_data->resting_potentials = resting_potentials;
      neuron_data->current_injections = current_injections;
      neuron_data->total_current_conductance = total_current_conductance;
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
                              frontend()->after_spike_reset_membrane_potentials_c,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));

      // Free tmp_last_spike_times
      free (tmp_last_spike_times);
      reset_current_injections();
    }
    
    void SpikingNeurons::reset_current_injections() {
      CudaSafeCall(cudaMemset(current_injections, 0.0f, sizeof(float)*(frontend()->total_number_of_neurons*frontend()->model->timestep_grouping)));
      CudaSafeCall(cudaMemset(total_current_conductance, 0.0f, sizeof(float)*(frontend()->total_number_of_neurons*frontend()->model->timestep_grouping)));
    }


    void SpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
    }


  } // ::Backend::CUDA
} // ::Backend
