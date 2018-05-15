// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/WeightDependentSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, WeightDependentSTDPPlasticity);

namespace Backend {
  namespace CUDA {
    WeightDependentSTDPPlasticity::~WeightDependentSTDPPlasticity() {
      CudaSafeCall(cudaFree(stdp_pre_memory_trace));
      CudaSafeCall(cudaFree(stdp_pre_memory_trace_update_time));
      CudaSafeCall(cudaFree(stdp_post_memory_trace));
      CudaSafeCall(cudaFree(activated_post_neuron_ids));
      CudaSafeCall(cudaFree(num_active_afferent_synapses));
      CudaSafeCall(cudaFree(active_afferent_synapse_counts));
      CudaSafeCall(cudaFree(weight_update_vals));
      CudaSafeCall(cudaFree(weight_update_times));

      if (h_stdp_trace)
        free(h_stdp_trace);
    }

    void WeightDependentSTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();
      
      CudaSafeCall(cudaMemset(num_active_afferent_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemcpy((void*)stdp_pre_memory_trace,
                              (void*)h_stdp_trace,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)stdp_pre_memory_trace_update_time,
                              (void*)h_stdp_trace,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)stdp_post_memory_trace,
                              (void*)h_stdp_trace,
                              sizeof(float)*frontend()->post_neuron_set.size(),
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)weight_update_vals,
                              (void*)h_stdp_trace,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)weight_update_times,
                              (void*)h_stdp_trace,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
    }

    void WeightDependentSTDPPlasticity::prepare() {
      STDPPlasticity::prepare();
      allocate_device_pointers();
    }

    void WeightDependentSTDPPlasticity::allocate_device_pointers() {
      // The following doesn't do anything in original code...
      // ::Backend::CUDA::STDPPlasticity::allocate_device_pointers();
      CudaSafeCall(cudaMalloc((void **)&stdp_pre_memory_trace, sizeof(float)*total_number_of_plastic_synapses));
      CudaSafeCall(cudaMalloc((void **)&stdp_pre_memory_trace_update_time, sizeof(float)*total_number_of_plastic_synapses));
      CudaSafeCall(cudaMalloc((void **)&stdp_post_memory_trace, sizeof(float)*frontend()->post_neuron_set.size()));
      CudaSafeCall(cudaMalloc((void **)&weight_update_vals, sizeof(float)*total_number_of_plastic_synapses));
      CudaSafeCall(cudaMalloc((void **)&weight_update_times, sizeof(float)*total_number_of_plastic_synapses));
      // Also allocate and set trace template
      h_stdp_trace = (float *)malloc( sizeof(float) * total_number_of_plastic_synapses);
      for (int i=0; i < total_number_of_plastic_synapses; i++){
        h_stdp_trace[i] = 0.0f;
      }
      // Allocating afferent space
      CudaSafeCall(cudaMalloc((void **)&num_activated_post_neurons, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&num_active_afferent_synapses, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&activated_post_neuron_ids, sizeof(int)*frontend()->post_neuron_set.size()));
      CudaSafeCall(cudaMalloc((void **)&active_afferent_synapse_counts, sizeof(int)*frontend()->post_neuron_set.size()));
    }

    void WeightDependentSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
      if (total_number_of_plastic_synapses > 0){ 
        update_active_plastic_elements(current_time_in_seconds, timestep);
        for (int g=0; g < frontend()->model->timestep_grouping; g++){
          CudaSafeCall(cudaMemset(num_activated_post_neurons, 0, sizeof(int)));
          CudaSafeCall(cudaMemset(num_active_afferent_synapses, 0, sizeof(int)));
          post_trace_update<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>(
              *(frontend()->stdp_params),
              post_neuron_set,
              neurons_backend->last_spike_time_of_each_neuron,
              stdp_post_memory_trace,
              num_activated_post_neurons,
              activated_post_neuron_ids,
              num_active_afferent_synapses,
              post_neuron_afferent_counts,
              (current_time_in_seconds + g*timestep),
              timestep,
              frontend()->post_neuron_set.size());
          CudaCheckError();
          
          int bufferloc = (int)(std::round(current_time_in_seconds / timestep) + g) % spike_buffer.buffer_size;
          pre_trace_update<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>(
              *(frontend()->stdp_params),
              spike_buffer,
              bufferloc,
              total_number_of_plastic_synapses,
              stdp_pre_memory_trace,
              stdp_pre_memory_trace_update_time,
              (current_time_in_seconds + g*timestep),
              timestep);
          CudaCheckError();


          on_pre<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>(
              *(frontend()->stdp_params),
              spike_buffer,
              bufferloc,
              total_number_of_plastic_synapses,
              synapses_backend->postsynaptic_neuron_indices,
              post_neuron_conversion,
              plastic_synapse_indices,
              stdp_post_memory_trace,
              synapses_backend->synaptic_efficacies_or_weights,
              weight_update_vals,
              weight_update_times,
              (current_time_in_seconds + g*timestep),
              timestep);
          CudaCheckError();

          CudaSafeCall(cudaMemcpy(
                &h_num_active_afferent_synapses,
                num_active_afferent_synapses,
                sizeof(int), cudaMemcpyDeviceToHost));
          int blocks_per_grid = ((h_num_active_afferent_synapses / synapses_backend->threads_per_block.x) + 1);
          if (blocks_per_grid > synapses_backend->max_num_blocks_per_grid)
            blocks_per_grid = synapses_backend->max_num_blocks_per_grid;
          on_post<<<blocks_per_grid, synapses_backend->threads_per_block.x>>>(
              *(frontend()->stdp_params),
              activated_post_neuron_ids,
              post_neuron_afferent_counts,
              post_neuron_afferent_totals,
              post_neuron_afferent_ids,
              num_active_afferent_synapses,
              total_number_of_plastic_synapses,
              plastic_synapse_indices,
              stdp_pre_memory_trace,
              stdp_pre_memory_trace_update_time,
              synapses_backend->synaptic_efficacies_or_weights,
              weight_update_vals,
              weight_update_times,
              (current_time_in_seconds + g*timestep),
              timestep);
          CudaCheckError();
        }

        /*
        ltp_and_ltd<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
          (synapses_backend->postsynaptic_neuron_indices,
           synapses_backend->time_of_last_spike_to_reach_synapse,
           neurons_backend->last_spike_time_of_each_neuron,
           synapses_backend->synaptic_efficacies_or_weights,
           stdp_pre_memory_trace,
           stdp_post_memory_trace,
           *(frontend()->stdp_params),
           timestep,
           frontend()->model->timestep_grouping,
           current_time_in_seconds,
           plastic_synapse_indices,
           total_number_of_plastic_synapses);
          */
        
      }
    }

    __global__ void post_trace_update(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        int* post_neuron_set,
        float* d_last_spike_time_of_each_neuron,
        float* stdp_post_memory_trace,
        int* num_activated_post_neurons,
        int* activated_post_neuron_ids,
        int* num_active_afferent_synapses,
        int* post_neuron_afferent_counts,
        float current_time_in_seconds,
        float timestep,
        size_t num_post_neurons)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all neurons
      while (indx < num_post_neurons) {
        int idx = post_neuron_set[indx];
        // Decay trace
        stdp_post_memory_trace[indx] *= expf(-(timestep / stdp_vars.tau_plus));;
        // If there was a spike, add to the trace
        if (abs(d_last_spike_time_of_each_neuron[idx] - current_time_in_seconds) < (0.5f*timestep)){
          float increment = stdp_vars.a_plus;
          stdp_post_memory_trace[indx] = stdp_vars.nearest_spike_only ? increment : (stdp_post_memory_trace[indx] + increment);
          // Add the set of afferent synapses to our list - to be used for on_post
          int pos = atomicAdd(&num_activated_post_neurons[0], 1);
          activated_post_neuron_ids[pos] = indx;
          atomicAdd(&num_active_afferent_synapses[0], post_neuron_afferent_counts[indx]);
        }
        indx += blockDim.x * gridDim.x;
      }
    }
    
    __global__ void pre_trace_update(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        device_circular_spike_buffer_struct spike_buffer,
        int bufferloc,
        int total_number_of_plastic_synapses,
        float* stdp_pre_memory_trace,
        float* stdp_pre_memory_trace_update_time,
        float current_time_in_seconds,
        float timestep)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < spike_buffer.time_buffer[bufferloc]) {
        int idx = spike_buffer.id_buffer[bufferloc*total_number_of_plastic_synapses + indx];
        // Decay trace
        if (stdp_vars.nearest_spike_only){
          stdp_pre_memory_trace[idx] = stdp_vars.a_minus;
        } else {
          stdp_pre_memory_trace[idx] *= 
            expf(-(current_time_in_seconds - stdp_pre_memory_trace_update_time[idx]) / stdp_vars.tau_minus);
          stdp_pre_memory_trace[idx] += stdp_vars.a_minus;
        }
        stdp_pre_memory_trace_update_time[idx] = current_time_in_seconds;
        indx += blockDim.x * gridDim.x;
      }
    }


    __global__ void on_pre(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        device_circular_spike_buffer_struct spike_buffer,
        int bufferloc,
        int total_number_of_plastic_synapses,
        int* postsynaptic_neuron_ids,
        int* post_neuron_conversion,
        int* plastic_synapses,
        float* stdp_post_memory_trace,
        float* synaptic_efficacies_or_weights,
        float* weight_update_vals,
        float* weight_update_times,
        float current_time_in_seconds,
        float timestep)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < spike_buffer.time_buffer[bufferloc]) {
        int syn_index = spike_buffer.id_buffer[bufferloc*total_number_of_plastic_synapses + indx];
        int idx = plastic_synapses[syn_index];
        int postsyn_id = postsynaptic_neuron_ids[idx];
        float stdp_post_memory_trace_val = stdp_post_memory_trace[post_neuron_conversion[postsyn_id]];
        // Carry out LTD:
        float old_synaptic_weight = synaptic_efficacies_or_weights[idx];
        float new_synaptic_weight = old_synaptic_weight - stdp_vars.lambda * stdp_vars.alpha * old_synaptic_weight * stdp_post_memory_trace_val;
        if (new_synaptic_weight >= 0.0f){
          synaptic_efficacies_or_weights[idx] = new_synaptic_weight;
          weight_update_vals[syn_index] = new_synaptic_weight - old_synaptic_weight;
          weight_update_times[syn_index] = current_time_in_seconds;
        }
        indx += blockDim.x * gridDim.x;
      }
    }


    __global__ void on_post(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        int* activated_post_neuron_ids,
        int* post_neuron_afferent_counts,
        int* post_neuron_afferent_totals,
        int* post_neuron_afferent_ids,
        int* num_active_afferent_synapses,
        int total_number_of_plastic_synapses,
        int* plastic_synapse_indices,
        float* stdp_pre_memory_trace,
        float* stdp_pre_memory_trace_update_time,
        float* synaptic_efficacies_or_weights,
        float* weight_update_vals,
        float* weight_update_times,
        float current_time_in_seconds,
        float timestep)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      int pos = 0;
      int idx = indx; 
      while (indx < num_active_afferent_synapses[0]) {
        // Get the correct current pre_neuron pos
        int synapse_count = post_neuron_afferent_counts[activated_post_neuron_ids[pos]];
        while (idx >= synapse_count){
          idx -= synapse_count;
          pos += 1;
          synapse_count = post_neuron_afferent_counts[activated_post_neuron_ids[pos]];
        }

        // Get the correct plastic synapse and trace
        int syn_index = post_neuron_afferent_ids[post_neuron_afferent_totals[activated_post_neuron_ids[pos]] + idx];
        float stdp_pre_memory_trace_val = stdp_pre_memory_trace[syn_index];
        stdp_pre_memory_trace_val *= 
            expf(-(current_time_in_seconds - stdp_pre_memory_trace_update_time[syn_index]) / stdp_vars.tau_minus);
        int synapse_id = plastic_synapse_indices[syn_index];
        bool weight_already_updated = fabs(weight_update_vals[syn_index] - current_time_in_seconds) < 0.5*timestep;
 

        // Carry out LTD:
        float old_synaptic_weight = synaptic_efficacies_or_weights[synapse_id];
        if (weight_already_updated)
          old_synaptic_weight -= weight_update_vals[syn_index];
        float new_synaptic_weight = old_synaptic_weight + stdp_vars.lambda * (stdp_vars.w_max - old_synaptic_weight) *stdp_pre_memory_trace_val;
        if (weight_already_updated)
          new_synaptic_weight += weight_update_vals[syn_index];
        synaptic_efficacies_or_weights[synapse_id] = new_synaptic_weight;

        idx += blockDim.x * gridDim.x;
        indx += blockDim.x * gridDim.x;
      }
    }
    /*
    __global__ void ltp_and_ltd (
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        float* stdp_post_memory_trace,
        float* stdp_pre_memory_trace,
        float* stdp_pre_memory_trace_update_time,
        float* synaptic_efficacies_or_weights,
        int total_number_of_plastic_synapses)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

    while (indx < ) {

        indx += blockDim.x * gridDim.x;
      }


    }


    __device__ void pre_trace_update(
        int synapse_index,
        float* pre_memory_trace,
        float a_minus,
        bool nearest_spike_only
        )
    {
      float increment = a_minus;
      *pre_memory_trace = nearest_spike_only ? a_minus
        *neuron_post_memory_trace = nearest_spike_only ? a_plus : (*neuron_post_memory_trace + a_plus);
      }
    }
    __global__ void pre_trace_update_and_on_pre(){}
    __global__ void on_post(){}

    __global__ void ltp_and_ltd
          (int* d_postsyns,
           float* d_time_of_last_spike_to_reach_synapse,
           float* d_last_spike_time_of_each_neuron,
           float* d_synaptic_efficacies_or_weights,
           float* stdp_pre_memory_trace,
           float* stdp_post_memory_trace,
           weightdependent_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses){
      // Global Index
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all neurons
      while (indx < total_number_of_plastic_synapses) {
        int idx = d_plastic_synapse_indices[indx];
        // First decay the memory trace (USING INDX FOR TRACE HERE AND BELOW)
        float stdp_pre_memory_trace_val = stdp_pre_memory_trace[indx];
        float stdp_post_memory_trace_val = stdp_post_memory_trace[indx];
        int postid = d_postsyns[idx];

        for (int g=0; g < timestep_grouping; g++){	
          stdp_post_memory_trace_val *= expf( - timestep / stdp_vars.tau_minus);
          stdp_pre_memory_trace_val *= expf(- timestep / stdp_vars.tau_plus);

          // First update the memory trace for every pre and post neuron
          if (fabs(d_time_of_last_spike_to_reach_synapse[idx] - (current_time_in_seconds + g*timestep)) < 0.5f*timestep){
            // Update the presynaptic memory trace
            stdp_pre_memory_trace_val += stdp_vars.a_plus;
            if (stdp_vars.nearest_spike_only)
              stdp_pre_memory_trace_val = stdp_vars.a_plus;
            // Carry out the necessary LTD
            float old_synaptic_weight = d_synaptic_efficacies_or_weights[idx];
            float new_synaptic_weight = old_synaptic_weight - stdp_vars.lambda * stdp_vars.alpha * old_synaptic_weight * stdp_post_memory_trace_val;
            if (new_synaptic_weight >= 0.0f)
              d_synaptic_efficacies_or_weights[idx] = new_synaptic_weight;
           }
          // Dealing with LTP
          if (fabs(d_last_spike_time_of_each_neuron[postid] - (current_time_in_seconds + g*timestep)) < 0.5f*timestep){
            stdp_post_memory_trace_val += stdp_vars.a_minus;
            if (stdp_vars.nearest_spike_only)
              stdp_post_memory_trace_val = stdp_vars.a_minus;
              // If output neuron just fired, do LTP
              d_synaptic_efficacies_or_weights[idx] += stdp_vars.lambda * (stdp_vars.w_max - d_synaptic_efficacies_or_weights[idx]) *stdp_pre_memory_trace_val;
          }
        }

        // Correctly set the trace values
        stdp_pre_memory_trace[indx] = stdp_pre_memory_trace_val;
        stdp_post_memory_trace[indx] = stdp_post_memory_trace_val;

        indx += blockDim.x * gridDim.x;
      }

    }

    */
    
  }
}
