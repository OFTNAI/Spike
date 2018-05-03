// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, ConductanceSpikingSynapses);

namespace Backend {
  namespace CUDA {
    // ConductanceSpikingSynapses Destructor
    ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
      CudaSafeCall(cudaFree(neuron_wise_conductance_trace));
      CudaSafeCall(cudaFree(d_synaptic_data));
      free(h_neuron_wise_conductance_trace);
    }

    void ConductanceSpikingSynapses::prepare() {
      SpikingSynapses::prepare();

      // Set up per neuron conductances
      conductance_trace_length = frontend()->neuron_pop_size*frontend()->num_syn_labels;
      h_neuron_wise_conductance_trace = (float*)realloc(h_neuron_wise_conductance_trace, conductance_trace_length*sizeof(float));
      for (int id = 0; id < conductance_trace_length; id++)
        h_neuron_wise_conductance_trace[id] = 0.0f;

      // Carry out remaining device actions
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();

      synaptic_data = new conductance_spiking_synapses_data_struct();
      memcpy(synaptic_data, (static_cast<ConductanceSpikingSynapses*>(this)->SpikingSynapses::synaptic_data), sizeof(spiking_synapses_data_struct));
      synaptic_data->decay_terms_tau_g = d_decay_terms_tau_g;
      synaptic_data->reversal_potentials_Vhat = d_reversal_potentials_Vhat;
      synaptic_data->neuron_wise_conductance_trace = neuron_wise_conductance_trace;
      CudaSafeCall(cudaMemcpy(
        d_synaptic_data,
        synaptic_data,
        sizeof(conductance_spiking_synapses_data_struct), cudaMemcpyHostToDevice));

    }

    void ConductanceSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
      CudaSafeCall(cudaMemcpy(
        neuron_wise_conductance_trace,
        h_neuron_wise_conductance_trace,
        sizeof(float)*conductance_trace_length, cudaMemcpyHostToDevice));

    }


    void ConductanceSpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&neuron_wise_conductance_trace, sizeof(float)*conductance_trace_length));
      CudaSafeCall(cudaMalloc((void **)&d_decay_terms_tau_g, sizeof(float)*frontend()->num_syn_labels));
      CudaSafeCall(cudaMalloc((void **)&d_reversal_potentials_Vhat, sizeof(float)*frontend()->num_syn_labels));
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data, sizeof(conductance_spiking_synapses_data_struct)));
    }

    void ConductanceSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(
        neuron_wise_conductance_trace,
        h_neuron_wise_conductance_trace,
        sizeof(float)*conductance_trace_length, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        d_decay_terms_tau_g,
        &(frontend()->decay_terms_tau_g[0]),
        sizeof(float)*frontend()->num_syn_labels, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        d_reversal_potentials_Vhat,
        &(frontend()->reversal_potentials_Vhat[0]),
        sizeof(float)*frontend()->num_syn_labels, cudaMemcpyHostToDevice));
    }



    /* STATE UPDATE */
    void ConductanceSpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {
      SpikingSynapses::state_update(neurons, input_neurons, current_time_in_seconds, timestep);
      
      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons->backend());
      assert(neurons_backend);
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(input_neurons->backend());
      assert(input_neurons_backend);


      //__constant__ pfunc dev_func_ptr = current_injection_kernel;
      pfunc host_pointer;
      //CudaSafeCall(cudaMemcpyFromSymbol(&host_pointer, "current_injection_kernel", sizeof(pfunc)));

      conductance_calculate_postsynaptic_current_injection_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
        host_pointer,
	d_synaptic_data,
	neurons_backend->d_neuron_data,
        d_decay_terms_tau_g,
        d_reversal_potentials_Vhat,
        frontend()->num_syn_labels,
        neuron_wise_conductance_trace,
        neuron_wise_input_update,
        neurons_backend->current_injections,
	neurons_backend->total_current_conductance,
        timestep,
	frontend()->model->timestep_grouping,
        neurons_backend->frontend()->total_number_of_neurons);
      CudaCheckError();

    }


    /* KERNELS BELOW */
    __device__ float current_injection_kernel(
        spiking_synapses_data_struct* in_synaptic_data,
	spiking_neurons_data_struct* neuron_data,
        float timestep,
        int timestep_grouping,
	int idx,
	int syn_label){
	
	conductance_spiking_synapses_data_struct* synaptic_data = (conductance_spiking_synapses_data_struct*) in_synaptic_data;
      
	int total_number_of_neurons =  neuron_data->total_number_of_neurons;
	// If we are on the first timestep group, reset the current and conductance values 
        /*
	if (g == 0){
            for (int g_prime = 0; g_prime < timestep_grouping; g_prime++){
	        neuron_data->current_injections[idx + g_prime*total_number_of_neurons] = 0.0f;
		synaptic_data->neuron_wise_conductance_trace[idx + g_prime*total_number_of_neurons] = 0.0f;
	    }
	} */

	// Update current injection and conductance	
//        for (int syn_label = 0; syn_label < synaptic_data->num_syn_labels; syn_label++){
          float decay_term_value = synaptic_data->decay_terms_tau_g[syn_label];
	  float decay_factor = expf(- timestep / decay_term_value);
	  float reversal_value = synaptic_data->reversal_potentials_Vhat[syn_label];
          float synaptic_conductance_g = synaptic_data->neuron_wise_conductance_trace[total_number_of_neurons*syn_label + idx];
	  for (int g=0; g < timestep_grouping; g++){
	  
          // Update the synaptic conductance
	  synaptic_conductance_g *= decay_factor;
	  synaptic_conductance_g += synaptic_data->neuron_wise_input_update[total_number_of_neurons*timestep_grouping*syn_label + g*total_number_of_neurons + idx];
	  // Reset the conductance update
	  synaptic_data->neuron_wise_input_update[total_number_of_neurons*timestep_grouping*syn_label + g*total_number_of_neurons + idx] = 0.0f;
	  // Set the currents and conductances -> Can we aggregate these?
          neuron_data->current_injections[idx + g*total_number_of_neurons] += synaptic_conductance_g * reversal_value;
          neuron_data->total_current_conductance[idx + g*total_number_of_neurons] += synaptic_conductance_g;
	  }
          synaptic_data->neuron_wise_conductance_trace[total_number_of_neurons*syn_label + idx] = synaptic_conductance_g;

//	}
	 return 0.0f;

    }

    __global__ void conductance_calculate_postsynaptic_current_injection_kernel(
	pfunc host_pointer,
        spiking_synapses_data_struct* synaptic_data,
	spiking_neurons_data_struct* neuron_data,
                  float* decay_term_values,
		  float* reversal_values,
                  int num_syn_labels,
                  float* neuron_wise_conductance_traces,
                  float* neuron_wise_input_update,
                  float* d_neurons_current_injections,
		  float* d_total_current_conductance,
                  float timestep,
		  int timestep_grouping,
                  size_t total_number_of_neurons){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {
	
	// First, resetting the current injection values
	for (int g=0; g < timestep_grouping; g++){
	  d_neurons_current_injections[idx + g*total_number_of_neurons] = 0.0f;
	  d_total_current_conductance[idx + g*total_number_of_neurons] = 0.0f;
	}
	/*
	// Updating current and conductance values
        for (int syn_label = 0; syn_label < num_syn_labels; syn_label++){
          float decay_term_value = decay_term_values[syn_label];
	  float decay_factor = expf(- timestep / decay_term_value);
	  float reversal_value = reversal_values[syn_label];
          float synaptic_conductance_g = neuron_wise_conductance_traces[total_number_of_neurons*syn_label + idx];
	  for (int g=0; g < timestep_grouping; g++){
            // Update the synaptic conductance
	    synaptic_conductance_g *= decay_factor;
	    synaptic_conductance_g += neuron_wise_input_update[total_number_of_neurons*timestep_grouping*syn_label + g*total_number_of_neurons + idx];
	    // Reset the conductance update
	    neuron_wise_input_update[total_number_of_neurons*timestep_grouping*syn_label + g*total_number_of_neurons + idx] = 0.0f;
	    // Set the currents and conductances -> Can we aggregate these?
            d_neurons_current_injections[idx + g*total_number_of_neurons] += synaptic_conductance_g * reversal_value;
            d_total_current_conductance[idx + g*total_number_of_neurons] += synaptic_conductance_g;
          }
	  // Set the conductance ready for the next timestep group
          neuron_wise_conductance_traces[total_number_of_neurons*syn_label + idx] = synaptic_conductance_g;
  	}
	*/
	for (int syn_label=0; syn_label < num_syn_labels; syn_label++){
	  current_injection_kernel(synaptic_data,
			neuron_data,
			timestep,
			timestep_grouping,
			idx,
			syn_label);	
	}

        idx += blockDim.x * gridDim.x;
      }
    }
   



  }
}
