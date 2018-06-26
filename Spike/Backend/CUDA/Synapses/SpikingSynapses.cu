// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingSynapses);

namespace Backend {
  namespace CUDA {

    __device__ injection_kernel spiking_device_kernel = spiking_current_injection_kernel;

    SpikingSynapses::SpikingSynapses() {
    }

    SpikingSynapses::~SpikingSynapses() {
      CudaSafeCall(cudaFree(delays));
      CudaSafeCall(cudaFree(d_syn_labels));
      CudaSafeCall(cudaFree(time_of_last_spike_to_reach_synapse));
      CudaSafeCall(cudaFree(group_indices));
      CudaSafeCall(cudaFree(num_active_synapses));
      CudaSafeCall(cudaFree(num_activated_neurons));
      CudaSafeCall(cudaFree(active_synapse_counts));
      CudaSafeCall(cudaFree(presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(neuron_inputs.circular_input_buffer));
      CudaSafeCall(cudaFree(d_synaptic_data));
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();

      // Set last spike times to -1000 so that the times do not affect current simulation.
      float *last_spike_to_reach_synapse = (float*)malloc(frontend()->total_number_of_synapses*sizeof(float));
      for (int i=0; i < frontend()->total_number_of_synapses; i++)
        last_spike_to_reach_synapse[i] = -1000.0f;

      CudaSafeCall(cudaMemcpy(time_of_last_spike_to_reach_synapse,
        last_spike_to_reach_synapse,
        frontend()->total_number_of_synapses*sizeof(float),
        cudaMemcpyHostToDevice));
      free(last_spike_to_reach_synapse);
      
      // Spike Buffer Resetting
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_activated_neurons, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(neuron_inputs.circular_input_buffer, 0.0f, sizeof(float)*neuron_inputs.temporal_buffersize*neuron_inputs.input_buffersize));
      CudaSafeCall(cudaMemset(neuron_inputs.bufferloc, 0, sizeof(int)));
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
      neuron_inputs.input_buffersize = frontend()->neuron_pop_size*frontend()->num_syn_labels;
      neuron_inputs.temporal_buffersize = buffersize;
      
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();

      synaptic_data = new spiking_synapses_data_struct();
      synaptic_data->num_syn_labels = frontend()->num_syn_labels;
      synaptic_data->neuron_inputs = neuron_inputs;
      CudaSafeCall(cudaMemcpy(d_synaptic_data,
        synaptic_data,
        sizeof(spiking_synapses_data_struct),
        cudaMemcpyHostToDevice));
         
    }

    void SpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&delays, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&d_syn_labels, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&time_of_last_spike_to_reach_synapse, sizeof(float)*frontend()->total_number_of_synapses));
      // Device pointers for spike buffer and active synapse/neuron storage
      CudaSafeCall(cudaMalloc((void **)&group_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&num_active_synapses, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&num_activated_neurons, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&active_synapse_counts, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&presynaptic_neuron_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data, sizeof(spiking_synapses_data_struct)));
      // Setting injection kernel
      CudaSafeCall(cudaMemcpyFromSymbol(
        &host_injection_kernel,
        spiking_device_kernel,
        sizeof(injection_kernel)));
      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.circular_input_buffer, sizeof(float)*neuron_inputs.temporal_buffersize*neuron_inputs.input_buffersize));
      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.bufferloc, sizeof(int)));
    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
        sizeof(int)*frontend()->total_number_of_synapses,
        cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        d_syn_labels,
        frontend()->syn_labels,
        sizeof(int)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
    }

    void SpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {
      
      // Calculate buffer location
      int bufferloc = (int)(std::round(current_time_in_seconds / timestep)) % buffersize;
      //synaptic_data->neuron_inputs = neuron_inputs;


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
      /*
      CudaSafeCall(cudaMemcpy(
          &h_num_active_synapses,
          num_active_synapses,
          sizeof(int), cudaMemcpyDeviceToHost));
      int blocks_per_grid = ((h_num_active_synapses / threads_per_block.x) + 1);
      if (blocks_per_grid > max_num_blocks_per_grid) blocks_per_grid = max_num_blocks_per_grid;
      */
      //activate_synapses<<<blocks_per_grid, threads_per_block>>>(
      activate_synapses<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
        neurons_backend->per_neuron_efferent_synapse_total,
        neurons_backend->per_neuron_efferent_synapse_indices,
        input_neurons_backend->per_neuron_efferent_synapse_total,
        input_neurons_backend->per_neuron_efferent_synapse_indices,
        bufferloc,
        buffersize,
        synaptic_data->neuron_inputs,
        postsynaptic_neuron_indices,
        synaptic_efficacies_or_weights,
        weight_scaling_constants,
        time_of_last_spike_to_reach_synapse,
        delays,
        d_syn_labels,
        timestep,
        current_time_in_seconds,
        frontend()->total_number_of_synapses,
        neurons_backend->frontend()->total_number_of_neurons,
        group_indices,
        frontend()->model->timestep_grouping,
        presynaptic_neuron_indices,
        active_synapse_counts,
        num_active_synapses);
      CudaCheckError();
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_activated_neurons, 0, sizeof(int)));
      
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
        indx += blockDim.x * gridDim.x;
      }
    }

    __global__ void activate_synapses(
        int* d_per_neuron_efferent_synapse_total,
        int* d_per_neuron_efferent_synapse_indices,
        int* d_per_input_neuron_efferent_synapse_total,
        int* d_per_input_neuron_efferent_synapse_indices,
        int bufferloc,
        int buffersize,
        neuron_inputs_struct neuron_inputs,
        int* postsynaptic_neuron_indices,
        float* synaptic_efficacies_or_weights,
        float* weight_scaling_constants,
        float* d_time_of_last_spike_to_reach_synapse,
        int* d_delays,
        int * d_syn_labels,
        float timestep,
        float current_time_in_seconds,
        int total_number_of_synapses,
        int total_number_of_neurons,
        int* group_indices,
        int timestep_grouping,
        int* presynaptic_neuron_indices,
        int* active_synapse_counts,
        int* num_active_synapses)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      if (indx == 0){
        neuron_inputs.bufferloc[0] = (bufferloc + timestep_grouping) % buffersize;
      }

      int pos = 0;
      int idx = indx;
      while (indx < num_active_synapses[0]) {
  
        int synapse_count = active_synapse_counts[pos];

        while(idx >= synapse_count){
          idx -= synapse_count;
          pos += 1;
          synapse_count = active_synapse_counts[pos];
        }

        int neuron = presynaptic_neuron_indices[pos];
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(neuron);
        int corr_idx = CORRECTED_PRESYNAPTIC_ID(neuron, presynaptic_is_input);

        int synapse_id = presynaptic_is_input ? d_per_input_neuron_efferent_synapse_indices[d_per_input_neuron_efferent_synapse_total[corr_idx] - idx - 1] : d_per_neuron_efferent_synapse_indices[d_per_neuron_efferent_synapse_total[corr_idx] - idx - 1];
        int postneuron = postsynaptic_neuron_indices[synapse_id];

        int targetloc = (bufferloc + d_delays[synapse_id] + group_indices[pos]) % buffersize;

        int syn_label = d_syn_labels[synapse_id];
        float weightinput = weight_scaling_constants[synapse_id]*synaptic_efficacies_or_weights[synapse_id];
        atomicAdd(&neuron_inputs.circular_input_buffer[targetloc*neuron_inputs.input_buffersize + syn_label*total_number_of_neurons + postneuron], weightinput);
        d_time_of_last_spike_to_reach_synapse[synapse_id] = current_time_in_seconds + (d_delays[synapse_id] + group_indices[pos])*timestep;

        indx += blockDim.x * gridDim.x;
        idx += blockDim.x * gridDim.x;
      }

    }

      __device__ float spiking_current_injection_kernel(
  spiking_synapses_data_struct* synaptic_data,
  spiking_neurons_data_struct* neuron_data,
  float current_membrane_voltage,
  float current_time_in_seconds,
  float timestep,
  float multiplication_to_volts,
  int timestep_grouping,
  int idx,
  int g){
        return 0.0f;
      };

  }
}
