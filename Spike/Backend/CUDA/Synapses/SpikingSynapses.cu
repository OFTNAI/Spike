// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingSynapses);

namespace Backend {
  namespace CUDA {

    __device__ injection_kernel spiking_device_kernel = spiking_current_injection_kernel;

    SpikingSynapses::~SpikingSynapses() {
      CudaSafeCall(cudaFree(delays));
      CudaSafeCall(cudaFree(d_syn_labels));
      CudaSafeCall(cudaFree(spikes_travelling_to_synapse));
      CudaSafeCall(cudaFree(time_of_last_spike_to_reach_synapse));
      CudaSafeCall(cudaFree(biological_conductance_scaling_constants_lambda));
      CudaSafeCall(cudaFree(circular_spikenum_buffer));
      CudaSafeCall(cudaFree(spikeid_buffer));
      CudaSafeCall(cudaFree(group_indices));
      CudaSafeCall(cudaFree(num_active_synapses));
      CudaSafeCall(cudaFree(num_activated_neurons));
      CudaSafeCall(cudaFree(active_synapse_counts));
      CudaSafeCall(cudaFree(presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(neuron_wise_input_update));
      CudaSafeCall(cudaFree(d_synaptic_data));
#ifdef CRAZY_DEBUG
      std::cout << "\n!!!!!!!!!!!!!!!!!!!!---BBBBBB---!!!!!!!!!!!!!!!!!!!\n";
#endif
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();

      CudaSafeCall(cudaMemset(spikes_travelling_to_synapse, 0,
                              sizeof(int)*frontend()->total_number_of_synapses));

      // Set last spike times to -1000 so that the times do not affect current simulation.
      float *last_spike_to_reach_synapse = (float*)malloc(frontend()->total_number_of_synapses*sizeof(float));
      for (int i=0; i < frontend()->total_number_of_synapses; i++)
        last_spike_to_reach_synapse[i] = -1000.0f;

      CudaSafeCall(cudaMemcpy(time_of_last_spike_to_reach_synapse,
                              last_spike_to_reach_synapse,
                              frontend()->total_number_of_synapses*sizeof(float),
                              cudaMemcpyHostToDevice));
      free(last_spike_to_reach_synapse);
      CudaSafeCall(cudaMemcpy(
        neuron_wise_input_update,
        h_neuron_wise_input_update,
        sizeof(float)*neuron_wise_input_length, cudaMemcpyHostToDevice));
      
      // Spike Buffer Resetting
      CudaSafeCall(cudaMemset(circular_spikenum_buffer, 0, sizeof(int)*buffersize));
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_activated_neurons, 0, sizeof(int)));
    }

    void SpikingSynapses::copy_weights_to_host() {
      CudaSafeCall(cudaMemcpy(frontend()->synaptic_efficacies_or_weights,
                              synaptic_efficacies_or_weights,
                              sizeof(float)*frontend()->total_number_of_synapses,
                              cudaMemcpyDeviceToHost));
    }

    void SpikingSynapses::prepare() {
      Synapses::prepare();
     
      // Extra buffer size for current time and extra to reset before last
      buffersize = frontend()->maximum_axonal_delay_in_timesteps + 2*frontend()->model->timestep_grouping + 1;
      
      neuron_wise_input_length = frontend()->neuron_pop_size*frontend()->num_syn_labels*frontend()->model->timestep_grouping;
      h_neuron_wise_input_update = (float*)realloc(h_neuron_wise_input_update, neuron_wise_input_length*sizeof(float));
      for (int id = 0; id < neuron_wise_input_length; id++)
        h_neuron_wise_input_update[id] = 0.0f;
      
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();

      synaptic_data = new spiking_synapses_data_struct();
      memcpy(synaptic_data, (static_cast<SpikingSynapses*>(this)->SpikingSynapses::synaptic_data), sizeof(synapses_data_struct));
      synaptic_data->neuron_wise_input_update = neuron_wise_input_update;
      synaptic_data->num_syn_labels = frontend()->num_syn_labels;
      CudaSafeCall(cudaMemcpy(d_synaptic_data,
                              synaptic_data,
                              sizeof(spiking_synapses_data_struct),
                              cudaMemcpyHostToDevice));
	       
    }

    void SpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&delays, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&d_syn_labels, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&spikes_travelling_to_synapse, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&time_of_last_spike_to_reach_synapse, sizeof(float)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&biological_conductance_scaling_constants_lambda, sizeof(float)*frontend()->total_number_of_synapses));
      // Device pointers for spike buffer and active synapse/neuron storage
      CudaSafeCall(cudaMalloc((void **)&circular_spikenum_buffer, sizeof(int)*buffersize));
      CudaSafeCall(cudaMalloc((void **)&spikeid_buffer, sizeof(int)*(buffersize * frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&group_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&num_active_synapses, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&num_activated_neurons, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&active_synapse_counts, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&presynaptic_neuron_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&neuron_wise_input_update, sizeof(float)*neuron_wise_input_length));
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data, sizeof(spiking_synapses_data_struct)));
      // Setting injection kernel
      CudaSafeCall(cudaMemcpyFromSymbol(
            &host_injection_kernel,
            spiking_device_kernel,
            sizeof(injection_kernel)));
    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(biological_conductance_scaling_constants_lambda,
        frontend()->biological_conductance_scaling_constants_lambda,
        sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        d_syn_labels,
        frontend()->syn_labels,
        sizeof(int)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        neuron_wise_input_update,
        h_neuron_wise_input_update,
        sizeof(float)*neuron_wise_input_length, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemset(circular_spikenum_buffer, 0, sizeof(int)*buffersize));
    }

    void SpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {
      
      // Calculate buffer location
      int bufferloc = (int)(std::round(current_time_in_seconds / timestep)) % buffersize;

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
                input_neurons_backend->per_neuron_efferent_synapse_count,
                neurons_backend->last_spike_time_of_each_neuron,
                input_neurons_backend->last_spike_time_of_each_neuron,
                current_time_in_seconds,
                timestep,
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
      
      move_spikes_towards_synapses_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
                  current_time_in_seconds,
		  circular_spikenum_buffer,
		  spikeid_buffer,
		  bufferloc,
		  buffersize,
		  frontend()->total_number_of_synapses,
		  time_of_last_spike_to_reach_synapse,
		  postsynaptic_neuron_indices,
		  neuron_wise_input_update,
		  d_syn_labels,
		  neurons_backend->frontend()->total_number_of_neurons,
		  synaptic_efficacies_or_weights,
		  biological_conductance_scaling_constants_lambda,
		  timestep,
		  frontend()->model->timestep_grouping);
      CudaCheckError();
    }
      
    
    __global__ void get_active_synapses_kernel(
		int* d_per_neuron_efferent_synapse_count,
		int* d_per_input_neuron_efferent_synapse_count,
                float* d_last_spike_time_of_each_neuron,
                float* d_last_spike_time_of_each_input_neuron,
                float current_time_in_seconds,
                float timestep,
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
         
	int groupindex = (int)lroundf((effecttime - current_time_in_seconds) / timestep);
        if (groupindex >= 0){
            int synapse_count = presynaptic_is_input ? d_per_input_neuron_efferent_synapse_count[corr_idx] : d_per_neuron_efferent_synapse_count[corr_idx];
            int pos = atomicAdd(&num_activated_neurons[0], 1);
	    atomicAdd(&num_active_synapses[0], synapse_count);
	    active_synapse_counts[pos] = synapse_count;
	    presynaptic_neuron_indices[pos] = idx;
	    group_indices[pos] = groupindex;
        }
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

    __global__ void move_spikes_towards_synapses_kernel(
                    float current_time_in_seconds,
		    int* circular_spikenum_buffer,
		    int* spikeid_buffer,
		    int bufferloc,
		    int buffersize,
		    int total_number_of_synapses,
                    float* d_time_of_last_spike_to_reach_synapse,
                    int* postsynaptic_neuron_indices,
                    float * neuron_wise_input_update,
                    int * d_syn_labels,
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
         int syn_label = d_syn_labels[idx];
         float synaptic_efficacy = d_biological_conductance_scaling_constants_lambda[idx] * d_synaptic_efficacies_or_weights[idx];
         atomicAdd(&neuron_wise_input_update[total_number_of_neurons*timestep_grouping*syn_label + g*total_number_of_neurons + postsynaptic_neuron_id], synaptic_efficacy);

          indx += blockDim.x * gridDim.x;
        }
        indx = threadIdx.x + blockIdx.x * blockDim.x;
      }
      __syncthreads();
    }
      __device__ float spiking_current_injection_kernel(
  spiking_synapses_data_struct* synaptic_data,
	spiking_neurons_data_struct* neuron_data,
  float current_membrane_voltage,
  float timestep,
  int timestep_grouping,
  int idx,
  int g){
	      return 0.0f;
      };

  }
}
