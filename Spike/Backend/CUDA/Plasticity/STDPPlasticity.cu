// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/STDPPlasticity.hpp"
#include <iostream>

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, STDPPlasticity);

namespace Backend {
  namespace CUDA {
    STDPPlasticity::~STDPPlasticity() {
      CudaSafeCall(cudaFree(plastic_synapse_indices));
      CudaSafeCall(cudaFree(post_neuron_set));
      CudaSafeCall(cudaFree(post_neuron_conversion));
      CudaSafeCall(cudaFree(pre_neuron_set));
      CudaSafeCall(cudaFree(pre_neuron_efferent_counts));
      //CudaSafeCall(cudaFree(pre_neuron_conversion));
      //CudaSafeCall(cudaFree(pre_input_neuron_conversion));
      CudaSafeCall(cudaFree(pre_neuron_efferent_totals));
      CudaSafeCall(cudaFree(pre_neuron_efferent_ids));
      CudaSafeCall(cudaFree(spike_buffer.time_buffer));
      CudaSafeCall(cudaFree(spike_buffer.id_buffer));
      CudaSafeCall(cudaFree(num_activated_pre_neurons));
      CudaSafeCall(cudaFree(num_active_efferent_synapses));
      CudaSafeCall(cudaFree(activated_pre_neuron_ids));
      CudaSafeCall(cudaFree(activated_pre_neuron_groupindices));

    }

    void STDPPlasticity::prepare() {

      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurs->backend());
      synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>
        (frontend()->syns->backend());

      // Get the correct ID
      int plasticity_id = frontend()->plasticity_rule_id;
      if (plasticity_id >= 0){
        total_number_of_plastic_synapses = frontend()->plastic_synapses.size();
      } else {
        total_number_of_plastic_synapses = 0;
      }

      // Sort out the spike buffer on the device
      spike_buffer.buffer_size = frontend()->model->spiking_synapses->maximum_axonal_delay_in_timesteps + 2*frontend()->model->timestep_grouping + 1;
      CudaSafeCall(cudaMalloc((void **)&spike_buffer.time_buffer, sizeof(int)*spike_buffer.buffer_size));
      CudaSafeCall(cudaMalloc((void **)&spike_buffer.id_buffer, sizeof(int)*(spike_buffer.buffer_size*total_number_of_plastic_synapses)));

      allocate_device_pointers();
    }

    void STDPPlasticity::allocate_device_pointers(){
      if (total_number_of_plastic_synapses > 0){
        CudaSafeCall(cudaMalloc((void **)&num_activated_pre_neurons, sizeof(int)));
        CudaSafeCall(cudaMalloc((void **)&num_active_efferent_synapses, sizeof(int)));
        CudaSafeCall(cudaMalloc((void **)&activated_pre_neuron_ids, sizeof(int)*frontend()->pre_neuron_set.size()));
        CudaSafeCall(cudaMalloc((void **)&activated_pre_neuron_groupindices, sizeof(int)*frontend()->pre_neuron_set.size()));
        CudaSafeCall(cudaMalloc((void **)&plastic_synapse_indices, sizeof(int)*total_number_of_plastic_synapses));
        CudaSafeCall(cudaMemcpy((void*)plastic_synapse_indices,
                                (void*)&(frontend()->plastic_synapses[0]),
                                sizeof(int)*total_number_of_plastic_synapses,
                                cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMalloc((void **)&post_neuron_set, sizeof(int)*frontend()->post_neuron_set.size()));
        CudaSafeCall(cudaMemcpy((void*)post_neuron_set,
                                (void*)&(frontend()->post_neuron_set[0]),
                                sizeof(int)*frontend()->post_neuron_set.size(),
                                cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMalloc((void **)&pre_neuron_set, sizeof(int)*frontend()->pre_neuron_set.size()));
        CudaSafeCall(cudaMemcpy((void*)pre_neuron_set,
                                (void*)&(frontend()->pre_neuron_set[0]),
                                sizeof(int)*frontend()->pre_neuron_set.size(),
                                cudaMemcpyHostToDevice));

        // Pre Neuron Efferents:
        CudaSafeCall(cudaMalloc((void **)&pre_neuron_efferent_totals, sizeof(int)*frontend()->pre_neuron_efferent_counts.size()));
        h_pre_neuron_efferent_totals.push_back(0);
        for (int index = 1; index < frontend()->pre_neuron_efferent_counts.size(); index++){
          h_pre_neuron_efferent_totals.push_back(h_pre_neuron_efferent_totals[index - 1] + frontend()->pre_neuron_efferent_counts[index - 1]);
        }
        CudaSafeCall(cudaMemcpy((void*)pre_neuron_efferent_totals,
                                (void*)&(h_pre_neuron_efferent_totals[0]),
                                sizeof(int)*h_pre_neuron_efferent_totals.size(),
                                cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMalloc((void **)&pre_neuron_efferent_counts, sizeof(int)*frontend()->pre_neuron_efferent_counts.size()));
        CudaSafeCall(cudaMemcpy((void*)pre_neuron_efferent_counts,
                                (void*)&(frontend()->pre_neuron_efferent_counts[0]),
                                sizeof(int)*frontend()->pre_neuron_efferent_counts.size(),
                                cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMalloc((void **)&pre_neuron_efferent_ids, sizeof(int)*total_number_of_plastic_synapses));
        for (int index = 0; index < frontend()->pre_neuron_efferent_counts.size(); index++){
          h_pre_neuron_efferent_ids.insert(h_pre_neuron_efferent_ids.end(), frontend()->pre_neuron_efferent_ids[index], frontend()->pre_neuron_efferent_ids[index] + frontend()->pre_neuron_efferent_counts[index]);
        }
        CudaSafeCall(cudaMemcpy((void*)pre_neuron_efferent_ids,
                                (void*)&(h_pre_neuron_efferent_ids[0]),
                                sizeof(int)*h_pre_neuron_efferent_ids.size(),
                                cudaMemcpyHostToDevice));

        //Post Neuron Afferents
        CudaSafeCall(cudaMalloc((void **)&post_neuron_afferent_totals, sizeof(int)*frontend()->post_neuron_afferent_counts.size()));
        h_post_neuron_afferent_totals.push_back(0);
        for (int index = 1; index < frontend()->post_neuron_afferent_counts.size(); index++){
          h_post_neuron_afferent_totals.push_back(h_post_neuron_afferent_totals[index - 1] + frontend()->post_neuron_afferent_counts[index - 1]);
        }
        CudaSafeCall(cudaMemcpy((void*)post_neuron_afferent_totals,
                                (void*)&(h_post_neuron_afferent_totals[0]),
                                sizeof(int)*h_post_neuron_afferent_totals.size(),
                                cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMalloc((void **)&post_neuron_afferent_counts, sizeof(int)*frontend()->post_neuron_afferent_counts.size()));
        CudaSafeCall(cudaMemcpy((void*)post_neuron_afferent_counts,
                                (void*)&(frontend()->post_neuron_afferent_counts[0]),
                                sizeof(int)*frontend()->post_neuron_afferent_counts.size(),
                                cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMalloc((void **)&post_neuron_afferent_ids, sizeof(int)*total_number_of_plastic_synapses));
        for (int index = 0; index < frontend()->post_neuron_afferent_counts.size(); index++){
          h_post_neuron_afferent_ids.insert(h_post_neuron_afferent_ids.end(), frontend()->post_neuron_afferent_ids[index], frontend()->post_neuron_afferent_ids[index] + frontend()->post_neuron_afferent_counts[index]);
        }
        CudaSafeCall(cudaMemcpy((void*)post_neuron_afferent_ids,
                                (void*)&(h_post_neuron_afferent_ids[0]),
                                sizeof(int)*h_post_neuron_afferent_ids.size(),
                                cudaMemcpyHostToDevice));
        /*
        CudaSafeCall(cudaMalloc((void **)&pre_neuron_conversion, sizeof(int)*frontend()->pre_neuron_conversion.size()));
        CudaSafeCall(cudaMemcpy((void*)pre_neuron_conversion,
                                (void*)&(frontend()->pre_neuron_conversion[0]),
                                sizeof(int)*frontend()->pre_neuron_conversion.size(),
                                cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMalloc((void **)&pre_input_neuron_conversion, sizeof(int)*frontend()->pre_neuron_conversion.size()));
        CudaSafeCall(cudaMemcpy((void*)pre_input_neuron_conversion,
                                (void*)&(frontend()->pre_input_neuron_conversion[0]),
                                sizeof(int)*frontend()->pre_input_neuron_conversion.size(),
                                cudaMemcpyHostToDevice));
                                */
        CudaSafeCall(cudaMalloc((void **)&post_neuron_conversion, sizeof(int)*frontend()->post_neuron_conversion.size()));
        CudaSafeCall(cudaMemcpy((void*)post_neuron_conversion,
                                (void*)&(frontend()->post_neuron_conversion[0]),
                                sizeof(int)*frontend()->post_neuron_conversion.size(),
                                cudaMemcpyHostToDevice));


      }
    }

    void STDPPlasticity::update_active_plastic_elements(float current_time_in_seconds, float timestep){
      // Use list of post-neurons to create a list that have spiked (to be dealt with)
      // Though this can also just be done by individual kernels ...

      // Use the list of pre-synaptic neurons with plastic efferents to:
      // Get activated neurons, place their synapses in a buffer
      // When the synapse time comes, place in an active list
      // Using SpikingSynapses as a model for this

      // Now each sub rule simply needs to check if the neurons have spiked and do "on_post"
      // Use the list of activated synapses to do "on_pre"
      // We can do on_pre first and always allow LTD primarily ... see how that goes
      int bufferloc = (int)(std::round(current_time_in_seconds / timestep)) % spike_buffer.buffer_size;
      
      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->model->spiking_neurons->backend());
      assert(neurons_backend);
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->model->input_spiking_neurons->backend());
      assert(input_neurons_backend);
      ::Backend::CUDA::SpikingSynapses* synapses_backend =
        dynamic_cast<::Backend::CUDA::SpikingSynapses*>(frontend()->model->spiking_synapses->backend());
      assert(synapses_backend);

      CudaSafeCall(cudaMemset(num_activated_pre_neurons, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_active_efferent_synapses, 0, sizeof(int)));
      
      get_active_preneurons_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>(
          pre_neuron_set,
          pre_neuron_efferent_counts,
          neurons_backend->last_spike_time_of_each_neuron,
          input_neurons_backend->last_spike_time_of_each_neuron,
          num_activated_pre_neurons,
          activated_pre_neuron_ids,
          activated_pre_neuron_groupindices,
          num_active_efferent_synapses,
          current_time_in_seconds,
          timestep,
          frontend()->pre_neuron_set.size());
      CudaCheckError();

      CudaSafeCall(cudaMemcpy(
            &h_num_active_synapses,
            num_active_efferent_synapses,
            sizeof(int), cudaMemcpyDeviceToHost));
      int blocks_per_grid = ((h_num_active_synapses / synapses_backend->threads_per_block.x) + 1);
      if (blocks_per_grid > synapses_backend->max_num_blocks_per_grid)
        blocks_per_grid = synapses_backend->max_num_blocks_per_grid;

      synapses_to_buffer_kernel<<<blocks_per_grid, synapses_backend->threads_per_block.x>>>(
        spike_buffer,
        bufferloc,
        synapses_backend->delays,
        total_number_of_plastic_synapses,
        frontend()->model->timestep_grouping,
        plastic_synapse_indices,
        pre_neuron_efferent_totals,
        pre_neuron_efferent_counts,
        pre_neuron_efferent_ids,
        activated_pre_neuron_ids,
        activated_pre_neuron_groupindices,
        num_active_efferent_synapses);
      CudaCheckError();
    
    }

    void STDPPlasticity::reset_state() {
      CudaSafeCall(cudaMemset(spike_buffer.time_buffer, 0, sizeof(int)*spike_buffer.buffer_size));
    }


    // KERNELS BELOW

    __global__ void get_active_preneurons_kernel(
        int* d_pre_neuron_set,
        int* d_pre_neuron_efferent_counts,
        float* d_last_spike_time_of_each_neuron,
        float* d_last_spike_time_of_each_input_neuron,
        int* d_num_activated_pre_neurons,
        int* d_activated_pre_neuron_ids,
        int* d_activated_pre_neuron_groupindices,
        int* d_num_active_efferent_synapses,
        float current_time_in_seconds,
        float timestep,
        size_t total_number_of_pre_neurons)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < total_number_of_pre_neurons){
        // Get actual neuron index
        int idx = d_pre_neuron_set[indx];
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(idx);
        int corr_idx = CORRECTED_PRESYNAPTIC_ID(idx, presynaptic_is_input);
        float effecttime = presynaptic_is_input ? d_last_spike_time_of_each_input_neuron[corr_idx] : d_last_spike_time_of_each_neuron[corr_idx];
        
        int group_index = (int)lroundf((effecttime - current_time_in_seconds) / timestep);
        if (group_index >= 0){
          int synapse_count = d_pre_neuron_efferent_counts[indx]; // INDX since converted list
          int pos = atomicAdd(&d_num_activated_pre_neurons[0], 1);
          atomicAdd(&d_num_active_efferent_synapses[0], synapse_count);
          d_activated_pre_neuron_ids[pos] = indx;
          d_activated_pre_neuron_groupindices[pos] = group_index;
        }
      
        indx += blockDim.x * gridDim.x;
      }
    }

    __global__ void synapses_to_buffer_kernel(
        device_circular_spike_buffer_struct spike_buffer,
        int bufferloc,
        int* d_delays,
        int total_number_of_plastic_synapses,
        int timestep_grouping,
        int* plastic_synapse_indices,
        int* d_pre_neuron_efferent_totals,
        int* d_pre_neuron_efferent_counts,
        int* d_pre_neuron_efferent_ids,
        int* d_activated_pre_neuron_ids,
        int* d_activated_pre_neuron_groupindices,
        int* d_num_active_efferent_synapses)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      // Reset the circular spike buffer if we need to
      if (indx == 0){
        for (int g=0; g < timestep_grouping; g++){
          int previd = (bufferloc + spike_buffer.buffer_size - g) % spike_buffer.buffer_size;
          spike_buffer.time_buffer[previd] = 0;
        }
      }
     
      int pos = 0;
      int idx = indx; 
      while (indx < d_num_active_efferent_synapses[0]){
        // Get the correct current pre_neuron pos
        int synapse_count = d_pre_neuron_efferent_counts[d_activated_pre_neuron_ids[pos]];
        while (idx >= synapse_count){
          idx -= synapse_count;
          pos += 1;
          synapse_count = d_pre_neuron_efferent_counts[d_activated_pre_neuron_ids[pos]];
        }

        int synapse_index = d_pre_neuron_efferent_ids[d_pre_neuron_efferent_totals[d_activated_pre_neuron_ids[pos]] + idx];
        int targetloc = (bufferloc + d_delays[plastic_synapse_indices[synapse_index]] + d_activated_pre_neuron_groupindices[pos]) % spike_buffer.buffer_size;
        int spikeloc = atomicAdd(&spike_buffer.time_buffer[targetloc], 1);
        spike_buffer.id_buffer[targetloc*total_number_of_plastic_synapses + spikeloc] = synapse_index;
      
        idx += blockDim.x * gridDim.x;
        indx += blockDim.x * gridDim.x;
      }
    }

  }
}
