// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, ConductanceSpikingSynapses);

namespace Backend {
  namespace CUDA {
    // ConductanceSpikingSynapses Destructor
    ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
      CudaSafeCall(cudaFree(biological_conductance_scaling_constants_lambda));
      CudaSafeCall(cudaFree(num_active_synapses));
      CudaSafeCall(cudaFreeHost(h_num_active_synapses));
      CudaSafeCall(cudaFreeHost(reset_deactivation));
      CudaSafeCall(cudaFree(active_synapse_indices));
      CudaSafeCall(cudaFree(num_after_deactivation));
      CudaSafeCall(cudaFree(synapse_switches));
      CudaSafeCall(cudaFree(synapse_decay_id));
      CudaSafeCall(cudaFree(neuron_wise_conductance_trace));
      CudaSafeCall(cudaFree(decay_term_values));
      free(h_decay_term_values);
      free(h_synapse_decay_id);
      free(h_neuron_wise_conductance_trace);
    }

    void ConductanceSpikingSynapses::prepare() {
      SpikingSynapses::prepare();

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
      conductance_trace_length = frontend()->neuron_pop_size*num_decay_terms;
      h_neuron_wise_conductance_trace = (float*)realloc(h_neuron_wise_conductance_trace, conductance_trace_length*sizeof(float));
      for (int id = 0; id < conductance_trace_length; id++)
        h_neuron_wise_conductance_trace[id] = 0.0f;

      // Carry out remaining device actions
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
    }

    void ConductanceSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_after_deactivation, 0, sizeof(int)));
      CudaSafeCall(cudaMemcpy(
        neuron_wise_conductance_trace,
        h_neuron_wise_conductance_trace,
        sizeof(float)*conductance_trace_length, cudaMemcpyHostToDevice));
      reset_deactivation[0] = false;
    }


    void ConductanceSpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&biological_conductance_scaling_constants_lambda, sizeof(float)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&active_synapse_indices, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&num_active_synapses, sizeof(int)));
      CudaSafeCall(cudaMallocHost((void **)&h_num_active_synapses, sizeof(int)));
      CudaSafeCall(cudaMallocHost((void **)&reset_deactivation, sizeof(bool)));
      CudaSafeCall(cudaMalloc((void **)&num_after_deactivation, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&synapse_switches, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&synapse_decay_id, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&neuron_wise_conductance_trace, sizeof(float)*conductance_trace_length));
      CudaSafeCall(cudaMalloc((void **)&decay_term_values, sizeof(float)*num_decay_terms));
      CudaSafeCall(cudaMalloc((void **)&reversal_values, sizeof(float)*num_decay_terms));
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
        decay_term_values,
        h_decay_term_values,
        sizeof(float)*num_decay_terms, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        reversal_values,
        h_reversal_values,
        sizeof(float)*num_decay_terms, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_after_deactivation, 0, sizeof(int)));
      reset_deactivation[0] = false;

    }



    /* STATE UPDATE */
    void ConductanceSpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {

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
                num_active_synapses,
                active_synapse_indices,
                num_after_deactivation,
                synapse_switches,
		reset_deactivation,
                timestep,
		input_neurons_backend->frontend()->total_number_of_neurons,
                (neurons_backend->frontend()->total_number_of_neurons + input_neurons_backend->frontend()->total_number_of_neurons),
     decay_term_values,
     reversal_values,
     num_decay_terms,
     synapse_decay_id,
     neuron_wise_conductance_trace,
     neurons_backend->current_injections,
     neurons_backend->membrane_potentials_v);
        CudaCheckError();
      
      // Setting up the custom block size for active synapses only:
      // Carry out the update every 10 timesteps. This timescale affects speed of the kernels not which syns
      if (fmod(current_time_in_seconds, 100.0*timestep) < timestep){
        // Copying to a pinned memory location (h_num_active_synapses) is much faster
        CudaSafeCall(cudaMemcpy(h_num_active_synapses, num_active_synapses, sizeof(int), cudaMemcpyDeviceToHost));
        active_syn_blocks_per_grid = dim3((h_num_active_synapses[0] + threads_per_block.x) /  threads_per_block.x);
        // Ensure we do not exceed the maximum number of efficient blocks
        if (active_syn_blocks_per_grid.x > number_of_synapse_blocks_per_grid.x)
          active_syn_blocks_per_grid = number_of_synapse_blocks_per_grid;
      }

      // Option for high fidelity. Ensures that a synapse can support multiple spikes
      if (neurons_backend->frontend()->high_fidelity_spike_flag){
        conductance_check_bitarray_for_presynaptic_neuron_spikes<<<active_syn_blocks_per_grid, threads_per_block>>>(
                  presynaptic_neuron_indices,
                  delays,
                  neurons_backend->bitarray_of_neuron_spikes,
                  input_neurons_backend->bitarray_of_neuron_spikes,
                  neurons_backend->frontend()->bitarray_length,
                  neurons_backend->frontend()->bitarray_maximum_axonal_delay_in_timesteps,
                  current_time_in_seconds,
                  timestep,
                  num_active_synapses,
                  active_synapse_indices,
                  time_of_last_spike_to_reach_synapse);
        CudaCheckError();
      } else {
        //CudaSafeCall(cudaMemcpy(h_num_active_synapses, num_after_deactivation, sizeof(int), cudaMemcpyDeviceToHost));
        if (reset_deactivation[0])
	 CudaSafeCall(cudaMemset(num_after_deactivation, 0, sizeof(int)));
        conductance_move_spikes_towards_synapses_kernel<<<active_syn_blocks_per_grid, threads_per_block>>>(
                  spikes_travelling_to_synapse,
                  current_time_in_seconds,
                  num_active_synapses,
                  active_synapse_indices,
                  num_after_deactivation,
                  synapse_switches,
		  reset_deactivation,
                  time_of_last_spike_to_reach_synapse,
		  postsynaptic_neuron_indices,
		  neuron_wise_conductance_trace,
		  synapse_decay_id,
		  neurons_backend->frontend()->total_number_of_neurons,
		  synaptic_efficacies_or_weights,
		  biological_conductance_scaling_constants_lambda,
                  timestep);
      }

     // conductance_calculate_postsynaptic_current_injection_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
     //   decay_term_values,
     //   reversal_values,
     //   num_decay_terms,
     //   synapse_decay_id,
     //   neuron_wise_conductance_trace,
     //   neurons_backend->current_injections,
     //   num_active_synapses,
     //   active_synapse_indices,
     //   neurons_backend->membrane_potentials_v, 
     //   timestep,
     //   neurons_backend->frontend()->total_number_of_neurons);

     // CudaCheckError();
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
                int* d_num_active_synapses,
                int* d_active_synapses,
                int* num_after_deactivation,
                int* synapse_switches,
		bool* reset_deactivation,
                float timestep,
		int num_input_neurons,
                size_t total_number_of_neurons,
                  float* decay_term_values,
                  float* reversal_values,
                  int num_decay_terms,
                  int* synapse_decay_values,
                  float* neuron_wise_conductance_traces,
                  float* d_neurons_current_injections,
                  float * d_membrane_potentials_v){

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < (total_number_of_neurons - num_input_neurons)){
	int idx = indx;

        float membrane_potential_v = d_membrane_potentials_v[idx];

        for (int decay_id = 0; decay_id < num_decay_terms; decay_id++){
	  if (decay_id == 0)
		  d_neurons_current_injections[idx] = 0.0f;
          float synaptic_conductance_g = neuron_wise_conductance_traces[idx + decay_id*total_number_of_neurons];
          // First decay the conductance values as required
          synaptic_conductance_g *= expf(- timestep / decay_term_values[decay_id]);
          neuron_wise_conductance_traces[idx + decay_id*total_number_of_neurons] = synaptic_conductance_g;
          d_neurons_current_injections[idx] += synaptic_conductance_g * (reversal_values[decay_id] - membrane_potential_v);
        }
	indx += blockDim.x * gridDim.x;
      }


      indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < total_number_of_neurons) {
   	int idx = indx - (num_input_neurons); 
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(idx);
	bool corr_idx = CORRECTED_PRESYNAPTIC_ID(idx, presynaptic_is_input);
        float effecttime = presynaptic_is_input ? d_last_spike_time_of_each_input_neuron[corr_idx] : d_last_spike_time_of_each_neuron[corr_idx];

        // Check if spike occurred within the last timestep    
        if (fabs(effecttime - current_time_in_seconds) < 0.5*timestep){
	  int synapse_count = presynaptic_is_input ? d_per_input_neuron_efferent_synapse_count[corr_idx] : d_per_neuron_efferent_synapse_count[corr_idx];  
          // For each of this neuron's efferent synapses
          for (int i = 0; i < synapse_count; i++){
            int synapse_id = presynaptic_is_input ? d_per_input_neuron_efferent_synapse_indices[d_per_input_neuron_efferent_synapse_total[corr_idx] - i - 1] : d_per_neuron_efferent_synapse_indices[d_per_neuron_efferent_synapse_total[corr_idx] - i - 1];
            // If this synapse is not active, make it active
            if (d_spikes_travelling_to_synapse[synapse_id] == 0) {
              int pos = atomicAdd(&num_after_deactivation[0], -1);
              pos -= 1;
              if (pos >= 0){
                d_active_synapses[synapse_switches[pos]] = synapse_id;
                d_spikes_travelling_to_synapse[synapse_id] = d_delays[synapse_id] + 1;
              } else {
		reset_deactivation[0] = true;
                pos = atomicAdd(&d_num_active_synapses[0], 1);
                d_active_synapses[pos] = synapse_id;  
                d_spikes_travelling_to_synapse[synapse_id] = d_delays[synapse_id] + 1;
              }
            }
          }
        }

        __syncthreads();
        indx += blockDim.x * gridDim.x;
      }
    }

    __global__ void conductance_calculate_postsynaptic_current_injection_kernel(
                  float* decay_term_values,
                  float* reversal_values,
                  int num_decay_terms,
                  int* synapse_decay_values,
                  float* neuron_wise_conductance_traces,
                  float* d_neurons_current_injections,
                  int* d_num_active_synapses,
                  int* d_active_synapses,
                  float * d_membrane_potentials_v,
                  float timestep,
                  size_t total_number_of_neurons){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

        float membrane_potential_v = d_membrane_potentials_v[idx];

        for (int decay_id = 0; decay_id < num_decay_terms; decay_id++){
	  if (decay_id == 0)
		  d_neurons_current_injections[idx] = 0.0f;
          float synaptic_conductance_g = neuron_wise_conductance_traces[idx + decay_id*total_number_of_neurons];
          // First decay the conductance values as required
          synaptic_conductance_g *= expf(- timestep / decay_term_values[decay_id]);
          neuron_wise_conductance_traces[idx + decay_id*total_number_of_neurons] = synaptic_conductance_g;
          d_neurons_current_injections[idx] += synaptic_conductance_g * (reversal_values[decay_id] - membrane_potential_v);
        }

        idx += blockDim.x * gridDim.x;

      }
    }

    __global__ void conductance_move_spikes_towards_synapses_kernel(
                    int* d_spikes_travelling_to_synapse,
                    float current_time_in_seconds,
                    int* d_num_active_synapses,
                    int* d_active_synapses,
                    int* num_after_deactivation,
                    int* synapse_switches,
		    bool* reset_deactivation,
                    float* d_time_of_last_spike_to_reach_synapse,
                                int* postsynaptic_neuron_indices,
                                float * neuron_wise_conductance_trace,
                                int * synaptic_decay_id,
                                int total_number_of_neurons,
                                float * d_synaptic_efficacies_or_weights,
                                float * d_biological_conductance_scaling_constants_lambda,
                    float timestep){

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < d_num_active_synapses[0]) {
        int idx = d_active_synapses[indx];
        if (idx < 0){
          indx += blockDim.x * gridDim.x;
          continue;
        }
        int timesteps_until_spike_reaches_synapse = d_spikes_travelling_to_synapse[idx];

        // If the spike is anything but zero (reset value) decrement it
        if (timesteps_until_spike_reaches_synapse != 0) 
          timesteps_until_spike_reaches_synapse -= 1;
        // If the spike is about to reach the synapse, set the spike time to next timestep
        if (timesteps_until_spike_reaches_synapse == 1){
            d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
            int postsynaptic_neuron_id = postsynaptic_neuron_indices[idx];
	    int trace_id = synaptic_decay_id[idx];
	    float synaptic_efficacy = d_biological_conductance_scaling_constants_lambda[idx] * d_synaptic_efficacies_or_weights[idx];
	    atomicAdd(&neuron_wise_conductance_trace[total_number_of_neurons*trace_id + postsynaptic_neuron_id], synaptic_efficacy);

        }
        // Given a spike injection window, check if the synapse is within it. If not, remove the synapse from active
        if (timesteps_until_spike_reaches_synapse == 0){
          int pos = atomicAdd(&num_after_deactivation[0], 1);
          synapse_switches[pos] = indx;
          d_active_synapses[indx] = -1;
	  reset_deactivation[0] = false;
        }

        d_spikes_travelling_to_synapse[idx] = timesteps_until_spike_reaches_synapse;

        indx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }

    __global__ void conductance_check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
                    int* d_delays,
                    unsigned char* d_bitarray_of_neuron_spikes,
                    unsigned char* d_input_neuron_bitarray_of_neuron_spikes,
                    int bitarray_length,
                    int bitarray_maximum_axonal_delay_in_timesteps,
                    float current_time_in_seconds,
                    float timestep,
                    int * d_num_active_synapses,
                    int * d_active_synapses,
                    float* d_time_of_last_spike_to_reach_synapse){

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < d_num_active_synapses[0]) {
        int idx = d_active_synapses[indx];

        int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
        int delay = d_delays[idx];

        // Get offset depending upon the current timestep
        int offset_index = ((int)(round(current_time_in_seconds / timestep)) % bitarray_maximum_axonal_delay_in_timesteps) - delay;
        offset_index = (offset_index < 0) ? (offset_index + bitarray_maximum_axonal_delay_in_timesteps) : offset_index;
        int offset_byte = offset_index / 8;
        int offset_bit_pos = offset_index - (8 * offset_byte);

        // Get the correct neuron index
        int neuron_index = CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input);

        // Check the spike
        int neuron_id_spike_store_start = neuron_index * bitarray_length;
        int check = 0;
        if (presynaptic_is_input){
          unsigned char byte = d_input_neuron_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          check = ((byte >> offset_bit_pos) & 1);
          if (check == 1){
            d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
          }
        } else {
          unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
          check = ((byte >> offset_bit_pos) & 1);
          if (check == 1){
            d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
          }
        }

        indx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }


    /* Kernel No Longer Used */
    __global__ void conductance_update_synaptic_conductances_kernel(
                                int* postsynaptic_neuron_indices,
                                float * neuron_wise_conductance_trace,
                                int * synaptic_decay_id,
                                int total_number_of_neurons,
                                float * d_synaptic_efficacies_or_weights,
                                float * d_time_of_last_spike_to_reach_synapse,
                                float * d_biological_conductance_scaling_constants_lambda,
                                int* d_num_active_synapses,
                                int* d_active_synapses,
                                float current_time_in_seconds) {

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < d_num_active_synapses[0]) {
        int idx = d_active_synapses[indx];
        if (idx < 0){
          indx += blockDim.x * gridDim.x;
          continue;
        }

        if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
          int postsynaptic_neuron_id = postsynaptic_neuron_indices[idx];
          int trace_id = synaptic_decay_id[idx];
          float synaptic_efficacy = d_biological_conductance_scaling_constants_lambda[idx] * d_synaptic_efficacies_or_weights[idx];
          atomicAdd(&neuron_wise_conductance_trace[total_number_of_neurons*trace_id + postsynaptic_neuron_id], synaptic_efficacy);
        }

        indx += blockDim.x * gridDim.x;
      }
      __syncthreads();

    }

  }
}
