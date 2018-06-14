// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/VoltageSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, VoltageSpikingSynapses);

namespace Backend {
  namespace CUDA {
    __device__ injection_kernel voltage_device_kernel = voltage_spiking_current_injection_kernel;

    // VoltageSpikingSynapses Destructor
    VoltageSpikingSynapses::~VoltageSpikingSynapses() {
    }

    void VoltageSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
    }

    void VoltageSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
    }


    void VoltageSpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMemcpyFromSymbol(
            &host_injection_kernel,
            voltage_device_kernel,
            sizeof(injection_kernel)));
    }

    void VoltageSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
    }



    /* STATE UPDATE */
    void VoltageSpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {
      SpikingSynapses::state_update(neurons, input_neurons, current_time_in_seconds, timestep);
    }


    /* KERNELS BELOW */
    __device__ float voltage_spiking_current_injection_kernel(
        spiking_synapses_data_struct* in_synaptic_data,
	      spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float current_membrane_voltage,
        float timestep,
        int timestep_grouping,
	      int idx,
	      int g){
      
      spiking_synapses_data_struct* synaptic_data = (spiking_synapses_data_struct*) in_synaptic_data;
        
	    int total_number_of_neurons =  neuron_data->total_number_of_neurons;
      float total_current = 0.0f;
      for (int syn_label = 0; syn_label < synaptic_data->num_syn_labels; syn_label++){
	      total_current += synaptic_data->neuron_wise_input_update[g*total_number_of_neurons*synaptic_data->num_syn_labels + syn_label*total_number_of_neurons + idx];
	      synaptic_data->neuron_wise_input_update[g*total_number_of_neurons*synaptic_data->num_syn_labels + syn_label*total_number_of_neurons + idx] = 0.0f;
      }


      // This is already in volts, no conversion necessary
      return total_current;
    }

  }
}
