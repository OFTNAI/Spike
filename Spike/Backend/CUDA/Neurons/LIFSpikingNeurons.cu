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
        (synapses_backend->d_synaptic_data,
         d_neuron_data,
         membrane_potentials_v,
         last_spike_time_of_each_neuron,
         membrane_resistances_R,
         membrane_time_constants_tau_m,
         resting_potentials,
         current_injections,
	 total_current_conductance,
	 thresholds_for_action_potential_spikes,
         frontend()->background_current,
         timestep,
	 frontend()->model->timestep_grouping,
         current_time_in_seconds,
         frontend()->refractory_period_in_seconds,
         frontend()->total_number_of_neurons);

      CudaCheckError();
    }
    
    __device__ float lif_current_injection_kernel(
        spiking_synapses_data_struct* in_synaptic_data,
	spiking_neurons_data_struct* neuron_data,
        float current_membrane_voltage,
        float timestep,
        int timestep_grouping,
	int idx,
	int g){
	
	conductance_spiking_synapses_data_struct* synaptic_data = (conductance_spiking_synapses_data_struct*) in_synaptic_data;
      
	int total_number_of_neurons =  neuron_data->total_number_of_neurons;
    float total_current = 0.0f;
        for (int syn_label = 0; syn_label < synaptic_data->num_syn_labels; syn_label++){
          float decay_term_value = synaptic_data->decay_terms_tau_g[syn_label];
	  float decay_factor = expf(- timestep / decay_term_value);
	  float reversal_value = synaptic_data->reversal_potentials_Vhat[syn_label];
          float synaptic_conductance_g = synaptic_data->neuron_wise_conductance_trace[total_number_of_neurons*syn_label + idx];
          // Update the synaptic conductance
	  synaptic_conductance_g *= decay_factor;
	  synaptic_conductance_g += synaptic_data->neuron_wise_input_update[total_number_of_neurons*timestep_grouping*syn_label + g*total_number_of_neurons + idx];
	  // Reset the conductance update
	  synaptic_data->neuron_wise_input_update[total_number_of_neurons*timestep_grouping*syn_label + g*total_number_of_neurons + idx] = 0.0f;
	  // Set the currents and conductances -> Can we aggregate these?
          //neuron_data->current_injections[idx + g*total_number_of_neurons] += synaptic_conductance_g * reversal_value;
          //neuron_data->total_current_conductance[idx + g*total_number_of_neurons] += synaptic_conductance_g;
          total_current += synaptic_conductance_g*(reversal_value - current_membrane_voltage);
          synaptic_data->neuron_wise_conductance_trace[total_number_of_neurons*syn_label + idx] = synaptic_conductance_g;

	}
	 return total_current;

    }

    
    __global__ void lif_update_membrane_potentials(
        spiking_synapses_data_struct* synaptic_data,
	spiking_neurons_data_struct* neuron_data,
  float *d_membrane_potentials_v,

                                                   float * d_last_spike_time_of_each_neuron,
                                                   float * d_membrane_resistances_R,
                                                   float * d_membrane_time_constants_tau_m,
                                                   float * d_resting_potentials,
                                                   float* d_current_injections,
                                                   float* d_total_current_conductance,
						   float* d_threshold_for_action_potential_spikes,
                                                   float background_current,
                                                   float timestep,
						   int timestep_grouping,
                                                   float current_time_in_seconds,
                                                   float refractory_period_in_seconds,
                                                   size_t total_number_of_neurons) {
      // // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

          float equation_constant = timestep / d_membrane_time_constants_tau_m[idx];
          float resting_potential_V0 = d_resting_potentials[idx];
          float temp_membrane_resistance_R = d_membrane_resistances_R[idx];
          float current_injection_Ii = d_current_injections[idx*timestep_grouping];
          float total_current_conductance = d_total_current_conductance[idx*timestep_grouping];
          float membrane_potential_Vi = d_membrane_potentials_v[idx];

  	  for (int g=0; g < timestep_grouping; g++){	  
              float current_injection_I = lif_current_injection_kernel(
                  synaptic_data,
                  neuron_data,
                  membrane_potential_Vi,
                  timestep,
                  timestep_grouping,
                  idx,
                  g);
            if (((current_time_in_seconds + g*timestep) - d_last_spike_time_of_each_neuron[idx]) >= refractory_period_in_seconds){
              current_injection_Ii = d_current_injections[g*total_number_of_neurons + idx];
              total_current_conductance = d_total_current_conductance[g*total_number_of_neurons + idx];
              //membrane_potential_Vi = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * (current_injection_Ii - total_current_conductance*membrane_potential_Vi)) + (1 - equation_constant) * membrane_potential_Vi + equation_constant * background_current;
              membrane_potential_Vi = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * current_injection_I) + (1 - equation_constant) * membrane_potential_Vi + equation_constant * background_current;
              
	  
	      // Finally check for a spike
	      if (membrane_potential_Vi >= d_threshold_for_action_potential_spikes[idx]){
	  	  d_last_spike_time_of_each_neuron[idx] = current_time_in_seconds + (g*timestep);
		  membrane_potential_Vi = d_resting_potentials[idx];
      //break;
		  continue;
	      }

	    }
	  }
          
	  d_membrane_potentials_v[idx] = membrane_potential_Vi;
	  
          idx += blockDim.x * gridDim.x;
        }
     } 


  } // namespace CUDA
} // namespace Backend
