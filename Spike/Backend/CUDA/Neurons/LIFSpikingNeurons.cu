// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, LIFSpikingNeurons);

namespace Backend {
  namespace CUDA {
    LIFSpikingNeurons::~LIFSpikingNeurons() {
      CudaSafeCall(cudaFree(membrane_time_constants_tau_m));
      CudaSafeCall(cudaFree(membrane_resistances_R));
    }

    void LIFSpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&membrane_time_constants_tau_m, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&membrane_resistances_R, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_neuron_data, sizeof(lif_spiking_neurons_data_struct)));
    }

    void LIFSpikingNeurons::copy_constants_to_device() {
      CudaSafeCall(cudaMemcpy(membrane_time_constants_tau_m,
                              frontend()->membrane_time_constants_tau_m,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(membrane_resistances_R,
                              frontend()->membrane_resistances_R,
                              sizeof(float)*frontend()->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
    }

    void LIFSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
      allocate_device_pointers();
      copy_constants_to_device();

      neuron_data = new lif_spiking_neurons_data_struct();
      memcpy(neuron_data, (static_cast<LIFSpikingNeurons*>(this)->SpikingNeurons::neuron_data), sizeof(spiking_neurons_data_struct));
      neuron_data->membrane_time_constants_tau_m = membrane_time_constants_tau_m;
      neuron_data->membrane_resistances_R = membrane_resistances_R;
      neuron_data->total_number_of_neurons = frontend()->total_number_of_neurons;
      CudaSafeCall(cudaMemcpy(d_neuron_data,
                              neuron_data,
                              sizeof(lif_spiking_neurons_data_struct),
                              cudaMemcpyHostToDevice));
    }

    void LIFSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }

    void LIFSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
      ::Backend::CUDA::ConductanceSpikingSynapses* synapses_backend =
        dynamic_cast<::Backend::CUDA::ConductanceSpikingSynapses*>(frontend()->model->spiking_synapses->backend());
      lif_update_membrane_potentials<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
        (synapses_backend->host_injection_kernel,
         synapses_backend->d_synaptic_data,
         d_neuron_data,
         frontend()->background_current,
         timestep,
	       frontend()->model->timestep_grouping,
         current_time_in_seconds,
         frontend()->refractory_period_in_seconds,
         frontend()->total_number_of_neurons);

      CudaCheckError();
    }
    /* KERNELS BELOW */
    __global__ void lif_update_membrane_potentials(
        injection_kernel current_injection_kernel,
        spiking_synapses_data_struct* synaptic_data,
	      spiking_neurons_data_struct* in_neuron_data,
        float background_current,
        float timestep,
				int timestep_grouping,
        float current_time_in_seconds,
        float refractory_period_in_seconds,
        size_t total_number_of_neurons) {
      // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

        lif_spiking_neurons_data_struct* neuron_data = (lif_spiking_neurons_data_struct*) in_neuron_data;
        float equation_constant = timestep / neuron_data->membrane_time_constants_tau_m[idx];
        float resting_potential_V0 = neuron_data->resting_potentials[idx];
        float temp_membrane_resistance_R = neuron_data->membrane_resistances_R[idx];
        float membrane_potential_Vi = neuron_data->membrane_potentials_v[idx];

    	  for (int g=0; g < timestep_grouping; g++){	  
            float current_injection_I = current_injection_kernel(
                  synaptic_data,
                  in_neuron_data,
                  membrane_potential_Vi,
                  timestep,
                  timestep_grouping,
                  idx,
                  g);
            if (((current_time_in_seconds + g*timestep) - neuron_data->last_spike_time_of_each_neuron[idx]) >= refractory_period_in_seconds){
              
              membrane_potential_Vi = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * current_injection_I) + (1 - equation_constant) * membrane_potential_Vi + equation_constant * background_current;
              
	  
	            // Finally check for a spike
	            if (membrane_potential_Vi >= neuron_data->thresholds_for_action_potential_spikes[idx]){
	  	          neuron_data->last_spike_time_of_each_neuron[idx] = current_time_in_seconds + (g*timestep);
		            membrane_potential_Vi = neuron_data->resting_potentials[idx];
                //break;
		            continue;
	      }

	    }
	  }
          
	  neuron_data->membrane_potentials_v[idx] = membrane_potential_Vi;
	  
          idx += blockDim.x * gridDim.x;
        }
     } 


  } // namespace CUDA
} // namespace Backend
