// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, ConductanceSpikingSynapses);

namespace Backend {
  namespace CUDA {
    // ConductanceSpikingSynapses Destructor
    ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
      CudaSafeCall(cudaFree(biological_conductance_scaling_constants_lambda));
      CudaSafeCall(cudaFree(synapse_decay_id));
      CudaSafeCall(cudaFree(neuron_wise_conductance_trace));
      CudaSafeCall(cudaFree(neuron_wise_conductance_update));
      CudaSafeCall(cudaFree(decay_term_values));
      CudaSafeCall(cudaFree(circular_spikenum_buffer));
      CudaSafeCall(cudaFree(spikeid_buffer));
      CudaSafeCall(cudaFree(group_indices));
      CudaSafeCall(cudaFree(num_active_synapses));
      CudaSafeCall(cudaFree(num_activated_neurons));
      CudaSafeCall(cudaFree(active_synapse_counts));
      CudaSafeCall(cudaFree(presynaptic_neuron_indices));
      free(h_decay_term_values);
      free(h_synapse_decay_id);
      free(h_neuron_wise_conductance_trace);
    }

    void ConductanceSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
      // Extra buffer size for current time and extra to reset before last
      buffersize = frontend()->maximum_axonal_delay_in_timesteps + 2*frontend()->model->timestep_grouping + 1;

      // Set up tau and reversal potential values and ids (Host-Side)
      h_synapse_decay_id = (int*)realloc(h_synapse_decay_id, frontend()->total_number_of_synapses*sizeof(int));
      // Prepare the tau synaptic conductance host-side vars
      for (int syn_id = 0; syn_id < frontend()->total_number_of_synapses; syn_id++){
        float tau_g = frontend()->decay_terms_tau_g[syn_id];
        float reversal_pot = frontend()->reversal_potentials_Vhat[syn_id];
        int id = -1;
        // Get Tau ID
        for (int i = 0; i < num_decay_terms; i++){
          // If this combination exists already, find it and assign the correct id
          if ((tau_g == h_decay_term_values[i]) && (reversal_pot == h_reversal_values[i]))
            id = i;       
        }
        if (id < 0){
	  // If this combination of tau/reversal potential doesn't exist, add it:
          num_decay_terms += 1;
          h_decay_term_values = (float*)realloc(h_decay_term_values, (num_decay_terms)*sizeof(float));
          h_reversal_values = (float*)realloc(h_reversal_values, (num_decay_terms)*sizeof(float));
          id = num_decay_terms - 1;
          h_decay_term_values[id] = tau_g;
          h_reversal_values[id] = reversal_pot;
        } 
        // Set this tau id
        h_synapse_decay_id[syn_id] = id;
      }
      // Set up per neuron conductances
      conductance_update_length = frontend()->neuron_pop_size*num_decay_terms*frontend()->model->timestep_grouping;
      conductance_trace_length = frontend()->neuron_pop_size*num_decay_terms;
      h_neuron_wise_conductance_trace = (float*)realloc(h_neuron_wise_conductance_trace, conductance_trace_length*sizeof(float));
      for (int id = 0; id < conductance_trace_length; id++)
        h_neuron_wise_conductance_trace[id] = 0.0f;
      h_neuron_wise_conductance_update = (float*)realloc(h_neuron_wise_conductance_update, conductance_update_length*sizeof(float));
      for (int id = 0; id < conductance_update_length; id++)
        h_neuron_wise_conductance_update[id] = 0.0f;

      // Carry out remaining device actions
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
    }

    void ConductanceSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
      CudaSafeCall(cudaMemcpy(
        neuron_wise_conductance_trace,
        h_neuron_wise_conductance_trace,
        sizeof(float)*conductance_trace_length, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        neuron_wise_conductance_update,
        h_neuron_wise_conductance_update,
        sizeof(float)*conductance_update_length, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemset(circular_spikenum_buffer, 0, sizeof(int)*buffersize));
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_activated_neurons, 0, sizeof(int)));

    }


    void ConductanceSpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&biological_conductance_scaling_constants_lambda, sizeof(float)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&synapse_decay_id, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&neuron_wise_conductance_trace, sizeof(float)*conductance_trace_length));
      CudaSafeCall(cudaMalloc((void **)&neuron_wise_conductance_update, sizeof(float)*conductance_update_length));
      CudaSafeCall(cudaMalloc((void **)&decay_term_values, sizeof(float)*num_decay_terms));
      CudaSafeCall(cudaMalloc((void **)&reversal_values, sizeof(float)*num_decay_terms));
      CudaSafeCall(cudaMalloc((void **)&circular_spikenum_buffer, sizeof(int)*buffersize));
      CudaSafeCall(cudaMalloc((void **)&spikeid_buffer, sizeof(int)*(buffersize * frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&group_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&num_active_synapses, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&num_activated_neurons, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&active_synapse_counts, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&presynaptic_neuron_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
    }

    void ConductanceSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(biological_conductance_scaling_constants_lambda,
        frontend()->biological_conductance_scaling_constants_lambda,
        sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        synapse_decay_id,
        h_synapse_decay_id,
        sizeof(int)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        neuron_wise_conductance_trace,
        h_neuron_wise_conductance_trace,
        sizeof(float)*conductance_trace_length, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        neuron_wise_conductance_update,
        h_neuron_wise_conductance_update,
        sizeof(float)*conductance_update_length, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        decay_term_values,
        h_decay_term_values,
        sizeof(float)*num_decay_terms, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        reversal_values,
        h_reversal_values,
        sizeof(float)*num_decay_terms, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemset(circular_spikenum_buffer, 0, sizeof(int)*buffersize));

    }



    /* STATE UPDATE */
    void ConductanceSpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {
     
      // Calculate buffer location
      int bufferloc = (int)(std::round(current_time_in_seconds / timestep)) % buffersize;

      // Setting up access to neuron backends
      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons->backend());
      assert(neurons_backend);
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(input_neurons->backend());
      assert(input_neurons_backend);

      // Steps for the synapses to cary out:
      // - Get the active synapses
      // - Update the delay sets based upon these
      // - Add any current where necessary (atomically)
      // - Deliver current to destination
      get_active_synapses_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
                neurons_backend->per_neuron_efferent_synapse_count,
                neurons_backend->per_neuron_efferent_synapse_total,
                neurons_backend->per_neuron_efferent_synapse_indices,
                input_neurons_backend->per_neuron_efferent_synapse_count,
                input_neurons_backend->per_neuron_efferent_synapse_total,
                input_neurons_backend->per_neuron_efferent_synapse_indices,
		delays,
                spikes_travelling_to_synapse,
                neurons_backend->last_spike_time_of_each_neuron,
                input_neurons_backend->last_spike_time_of_each_neuron,
                current_time_in_seconds,
                timestep,
		frontend()->model->timestep_grouping,
		input_neurons_backend->frontend()->total_number_of_neurons,
		group_indices,
		num_active_synapses,
		num_activated_neurons,
		active_synapse_counts,
		presynaptic_neuron_indices,
                (neurons_backend->frontend()->total_number_of_neurons + input_neurons_backend->frontend()->total_number_of_neurons));
      CudaCheckError();
      
      CudaSafeCall(cudaMemcpy(
        &h_num_active_synapses,
        num_active_synapses,
        sizeof(int), cudaMemcpyDeviceToHost));
      int blocks_per_grid = ((h_num_active_synapses / threads_per_block.x) + 1);
      if (blocks_per_grid > max_num_blocks_per_grid) blocks_per_grid = max_num_blocks_per_grid;
      //activate_synapses<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
      activate_synapses<<<blocks_per_grid, threads_per_block>>>(
                neurons_backend->per_neuron_efferent_synapse_total,
                neurons_backend->per_neuron_efferent_synapse_indices,
                input_neurons_backend->per_neuron_efferent_synapse_total,
                input_neurons_backend->per_neuron_efferent_synapse_indices,
		circular_spikenum_buffer,
		spikeid_buffer,
		bufferloc,
		buffersize,
		delays,
		frontend()->total_number_of_synapses,
		group_indices,
		frontend()->model->timestep_grouping,
		presynaptic_neuron_indices,
		active_synapse_counts,
		num_active_synapses);
      CudaCheckError();
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_activated_neurons, 0, sizeof(int)));
      
      conductance_move_spikes_towards_synapses_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
                  spikes_travelling_to_synapse,
                  current_time_in_seconds,
		  circular_spikenum_buffer,
		  spikeid_buffer,
		  bufferloc,
		  buffersize,
		  frontend()->total_number_of_synapses,
                  time_of_last_spike_to_reach_synapse,
		  postsynaptic_neuron_indices,
		  neuron_wise_conductance_update,
		  synapse_decay_id,
		  neurons_backend->frontend()->total_number_of_neurons,
		  synaptic_efficacies_or_weights,
		  biological_conductance_scaling_constants_lambda,
                  timestep,
		  frontend()->model->timestep_grouping);
      CudaCheckError();

      conductance_calculate_postsynaptic_current_injection_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
        decay_term_values,
        reversal_values,
        num_decay_terms,
        synapse_decay_id,
        neuron_wise_conductance_trace,
        neuron_wise_conductance_update,
        neurons_backend->current_injections,
	neurons_backend->total_current_conductance,
        timestep,
	frontend()->model->timestep_grouping,
        neurons_backend->frontend()->total_number_of_neurons);

    }


    /* KERNELS BELOW */
    __global__ void get_active_synapses_kernel(
		int* d_per_neuron_efferent_synapse_count,
        	int* d_per_neuron_efferent_synapse_total,
                int* d_per_neuron_efferent_synapse_indices,
		int* d_per_input_neuron_efferent_synapse_count,
        	int* d_per_input_neuron_efferent_synapse_total,
                int* d_per_input_neuron_efferent_synapse_indices,
		int* d_delays,
                int* d_spikes_travelling_to_synapse,
                float* d_last_spike_time_of_each_neuron,
                float* d_last_spike_time_of_each_input_neuron,
                float current_time_in_seconds,
                float timestep,
		int timestep_grouping,
		int num_input_neurons,
		int* group_indices,
		int* num_active_synapses,
		int* num_activated_neurons,
		int* active_synapse_counts,
		int* presynaptic_neuron_indices,
                size_t total_number_of_neurons) {

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < total_number_of_neurons) {
		
   	int idx = indx - (num_input_neurons); 
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(idx);
	int corr_idx = CORRECTED_PRESYNAPTIC_ID(idx, presynaptic_is_input);
        float effecttime = presynaptic_is_input ? d_last_spike_time_of_each_input_neuron[corr_idx] : d_last_spike_time_of_each_neuron[corr_idx];

        // Check if spike occurred within coming timestep group
	//for (int g=0; g < timestep_grouping; g++){
          //if (fabs(effecttime - (current_time_in_seconds + g*timestep)) < (0.5*timestep)){
          int groupindex = (int)lroundf((effecttime - current_time_in_seconds) / timestep);
          //if (fabs(effecttime - (current_time_in_seconds + g*timestep)) < (0.5*timestep)){
          if (groupindex >= 0){
            int synapse_count = presynaptic_is_input ? d_per_input_neuron_efferent_synapse_count[corr_idx] : d_per_neuron_efferent_synapse_count[corr_idx];
            int pos = atomicAdd(&num_activated_neurons[0], 1);
	    atomicAdd(&num_active_synapses[0], synapse_count);
	    active_synapse_counts[pos] = synapse_count;
	    presynaptic_neuron_indices[pos] = idx;
	    group_indices[pos] = groupindex;
            // Store group index, synapse count, is_input?
            // Accummulate synapse_count (Needs to be used for the kernel size
	    // Buffer size etc can be dealt with later


            // For each of this neuron's efferent synapses
	    /*
            for (int i = 0; i < synapse_count; i++){
              int synapse_id = presynaptic_is_input ? d_per_input_neuron_efferent_synapse_indices[d_per_input_neuron_efferent_synapse_total[corr_idx] - i - 1] : d_per_neuron_efferent_synapse_indices[d_per_neuron_efferent_synapse_total[corr_idx] - i - 1];
              // If this synapse is not active, make it active
	      int targetloc = (bufferloc + d_delays[synapse_id] + groupindex) % buffersize;
	      int pos = atomicAdd(&circular_spikenum_buffer[targetloc], 1);
	      spikeid_buffer[targetloc*total_number_of_synapses + pos] = synapse_id;
            }
	    */
	    // If we found a spike, there is no chance this neuron spikes again:
	    //break;
          }
	//}

        __syncthreads();
        indx += blockDim.x * gridDim.x;
      }
    }

    __global__ void activate_synapses
		(int* d_per_neuron_efferent_synapse_total,
                int* d_per_neuron_efferent_synapse_indices,
        	int* d_per_input_neuron_efferent_synapse_total,
                int* d_per_input_neuron_efferent_synapse_indices,
		int* circular_spikenum_buffer,
		int* spikeid_buffer,
		int bufferloc,
		int buffersize,
                int* d_delays,
		int total_number_of_synapses,
		int* group_indices,
		int timestep_grouping,
		int* presynaptic_neuron_indices,
		int* active_synapse_counts,
		int* num_active_synapses)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
	if (indx == 0){
	  for (int g=0; g < timestep_grouping; g++){
	    int previd = (bufferloc + g) % buffersize;
	    circular_spikenum_buffer[previd] = 0;
	  }
	}
      while (indx < num_active_synapses[0]) {
	int idx = indx;
	int pos = 0;
	int synapse_count = active_synapse_counts[pos];


	while(idx >= synapse_count){
	  idx -= synapse_count;
	  synapse_count = active_synapse_counts[++pos];
	}

	int neuron = presynaptic_neuron_indices[pos];
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(neuron);
	int corr_idx = CORRECTED_PRESYNAPTIC_ID(neuron, presynaptic_is_input);

        int synapse_id = presynaptic_is_input ? d_per_input_neuron_efferent_synapse_indices[d_per_input_neuron_efferent_synapse_total[corr_idx] - idx - 1] : d_per_neuron_efferent_synapse_indices[d_per_neuron_efferent_synapse_total[corr_idx] - idx - 1];
	int targetloc = (bufferloc + d_delays[synapse_id] + group_indices[pos]) % buffersize;
	int bufloc = atomicAdd(&circular_spikenum_buffer[targetloc], 1);
	spikeid_buffer[targetloc*total_number_of_synapses + bufloc] = synapse_id;

        indx += blockDim.x * gridDim.x;
      }

    }

    __global__ void conductance_move_spikes_towards_synapses_kernel(
                    int* d_spikes_travelling_to_synapse,
                    float current_time_in_seconds,
		    int* circular_spikenum_buffer,
		    int* spikeid_buffer,
		    int bufferloc,
		    int buffersize,
		    int total_number_of_synapses,
                    float* d_time_of_last_spike_to_reach_synapse,
                                int* postsynaptic_neuron_indices,
                                float * neuron_wise_conductance_update,
                                int * synaptic_decay_id,
                                int total_number_of_neurons,
                                float * d_synaptic_efficacies_or_weights,
                                float * d_biological_conductance_scaling_constants_lambda,
                    float timestep,
		    int timestep_grouping){

      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      for (int g=0; g < timestep_grouping; g++){
	int tmpbufferloc = (bufferloc + timestep_grouping + g) % buffersize;
        while (indx < circular_spikenum_buffer[tmpbufferloc]) {
         int idx = spikeid_buffer[tmpbufferloc*total_number_of_synapses + indx];
       
         // Update Synapses
         d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds + (timestep_grouping + g)*timestep;
         int postsynaptic_neuron_id = postsynaptic_neuron_indices[idx];
         int trace_id = synaptic_decay_id[idx];
         float synaptic_efficacy = d_biological_conductance_scaling_constants_lambda[idx] * d_synaptic_efficacies_or_weights[idx];
         atomicAdd(&neuron_wise_conductance_update[total_number_of_neurons*timestep_grouping*trace_id + postsynaptic_neuron_id*timestep_grouping + g], synaptic_efficacy);

          indx += blockDim.x * gridDim.x;
        }
        indx = threadIdx.x + blockIdx.x * blockDim.x;
      }
      __syncthreads();
    }
    
   
    __global__ void conductance_calculate_postsynaptic_current_injection_kernel(
                  float* decay_term_values, float* reversal_values,
                  int num_decay_terms,
                  int* synapse_decay_values,
                  float* neuron_wise_conductance_traces,
                  float* neuron_wise_conductance_update,
                  float* d_neurons_current_injections,
		  float* d_total_current_conductance,
                  float timestep,
		  int timestep_grouping,
                  size_t total_number_of_neurons){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {
	// First, resetting the current injection values
	for (int g=0; g < timestep_grouping; g++){
	  d_neurons_current_injections[idx*timestep_grouping + g] = 0.0f;
	  d_total_current_conductance[idx*timestep_grouping + g] = 0.0f;
	}
	// Updating current and conductance values
        for (int decay_id = 0; decay_id < num_decay_terms; decay_id++){
          float decay_term_value = decay_term_values[decay_id];
	  float decay_factor = expf(- timestep / decay_term_value);
	  float reversal_value = reversal_values[decay_id];
          float synaptic_conductance_g = neuron_wise_conductance_traces[total_number_of_neurons*decay_id + idx];
	  for (int g=0; g < timestep_grouping; g++){
            // Update the synaptic conductance
	    synaptic_conductance_g *= decay_factor;
	    synaptic_conductance_g += neuron_wise_conductance_update[total_number_of_neurons*timestep_grouping*decay_id + idx*timestep_grouping + g];
	    // Reset the conductance update
	    neuron_wise_conductance_update[total_number_of_neurons*timestep_grouping*decay_id + idx*timestep_grouping + g] = 0.0f;
	    // Set the currents and conductances -> Can we aggregate these?
            d_neurons_current_injections[idx*timestep_grouping + g] += synaptic_conductance_g * reversal_value;
            d_total_current_conductance[idx*timestep_grouping + g] += synaptic_conductance_g;
          }
	  // Set the conductance ready for the next timestep group
          neuron_wise_conductance_traces[total_number_of_neurons*decay_id + idx] = synaptic_conductance_g;
  	}

        idx += blockDim.x * gridDim.x;
      }
    }
   



  }
}
