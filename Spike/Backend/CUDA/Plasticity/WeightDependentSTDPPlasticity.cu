// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/WeightDependentSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, WeightDependentSTDPPlasticity);

namespace Backend {
  namespace CUDA {
    WeightDependentSTDPPlasticity::~WeightDependentSTDPPlasticity() {
      CudaSafeCall(cudaFree(stdp_pre_memory_trace));
      CudaSafeCall(cudaFree(stdp_post_memory_trace));
      if (h_stdp_trace)
        free(h_stdp_trace);
    }

    void WeightDependentSTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();
      
      CudaSafeCall(cudaMemcpy((void*)stdp_pre_memory_trace,
                              (void*)h_stdp_trace,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)stdp_post_memory_trace,
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
      CudaSafeCall(cudaMalloc((void **)&stdp_post_memory_trace, sizeof(float)*total_number_of_plastic_synapses));
      // Also allocate and set trace template
      h_stdp_trace = (float *)malloc( sizeof(float) * total_number_of_plastic_synapses);
      for (int i=0; i < total_number_of_plastic_synapses; i++){
        h_stdp_trace[i] = 0.0f;
      }
    }

    void WeightDependentSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
        ltp_and_ltd<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
          (synapses_backend->postsynaptic_neuron_indices,
           synapses_backend->presynaptic_neuron_indices,
           synapses_backend->delays,
           neurons_backend->d_neuron_data,
           input_neurons_backend->d_neuron_data,
           synapses_backend->synaptic_efficacies_or_weights,
           stdp_pre_memory_trace,
           stdp_post_memory_trace,
           frontend()->stdp_params->tau_minus,//expf(- timestep / frontend()->stdp_params->tau_minus),
           frontend()->stdp_params->tau_plus,//expf(- timestep / frontend()->stdp_params->tau_plus),
           *(frontend()->stdp_params),
           timestep,
           frontend()->model->timestep_grouping,
           current_time_in_seconds,
           plastic_synapse_indices,
           total_number_of_plastic_synapses);
          CudaCheckError();
    }


    __global__ void ltp_and_ltd
          (int* d_postsyns,
           int* d_presyns,
           int* d_syndelays,
           spiking_neurons_data_struct* neuron_data,
           spiking_neurons_data_struct* input_neuron_data,
           float* d_synaptic_efficacies_or_weights,
           float* stdp_pre_memory_trace,
           float* stdp_post_memory_trace,
           float post_decay,
           float pre_decay,
           weightdependent_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses){
      // Global Index
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      int bufsize = input_neuron_data->neuron_spike_time_bitbuffer_bytesize[0];

      // Running though all neurons
      while (indx < total_number_of_plastic_synapses) {
        int idx = d_plastic_synapse_indices[indx];

        // Getting synapse details
        float stdp_pre_memory_trace_val = stdp_pre_memory_trace[indx];
        float stdp_post_memory_trace_val = stdp_post_memory_trace[indx];
        int postid = d_postsyns[idx];
        int preid = d_presyns[idx];
        float old_synaptic_weight = d_synaptic_efficacies_or_weights[idx];
        float new_synaptic_weight = old_synaptic_weight;

        // Correcting for input vs output neuron types
        bool is_input = PRESYNAPTIC_IS_INPUT(preid);
        int corr_preid = CORRECTED_PRESYNAPTIC_ID(preid, is_input);
        uint8_t* pre_bitbuffer = is_input ? input_neuron_data->neuron_spike_time_bitbuffer : neuron_data->neuron_spike_time_bitbuffer;
        float* pre_last_spike_times = is_input ? input_neuron_data->last_spike_time_of_each_neuron : neuron_data->last_spike_time_of_each_neuron;



        //int pre_spike_g = -1;
        int pre_spike_g = ((int)roundf((pre_last_spike_times[corr_preid] - current_time_in_seconds) / timestep));
        int post_spike_g = ((int)roundf((neuron_data->last_spike_time_of_each_neuron[postid] - current_time_in_seconds) / timestep));
        if (pre_spike_g >= timestep_grouping)
          pre_spike_g *= -1;
        /*
        for (int g=0; g < timestep_grouping; g++){
        // Looping over timesteps
        int postbitloc = ((int)roundf(current_time_in_seconds / timestep) + g) % (bufsize*8);
        int prebitloc = postbitloc - d_syndelays[idx];
        prebitloc = (prebitloc < 0) ? (bufsize*8 + prebitloc) : prebitloc;
          if (pre_bitbuffer[corr_preid*bufsize + (prebitloc / 8)] & (1 << (prebitloc % 8)))
            pre_spike_g = g;
          //if (neuron_data->neuron_spike_time_bitbuffer[postid*bufsize + (postbitloc / 8)] & (1 << (postbitloc % 8)))
            //post_spike_g = g;
        }*/

        stdp_post_memory_trace_val *= expf(-(timestep_grouping*timestep) / post_decay);
        stdp_pre_memory_trace_val *= expf(-(timestep_grouping*timestep) / pre_decay);

        // Change this if nearest only
        stdp_post_memory_trace_val += (post_spike_g >= 0) ? stdp_vars.a_minus*expf(-((timestep_grouping - post_spike_g)*timestep) / post_decay) : 0.0f;
        stdp_pre_memory_trace_val += (pre_spike_g >= 0) ? stdp_vars.a_plus*expf(-((timestep_grouping - pre_spike_g)*timestep) / pre_decay) : 0.0f;
          
        float syn_update_val = 0.0f; 
        //old_synaptic_weight = new_synaptic_weight;
        // OnPre Weight Update
        if (pre_spike_g >= 0){
          float temp_post_trace = stdp_post_memory_trace_val;
          temp_post_trace += (post_spike_g > pre_spike_g) ? -stdp_vars.a_minus*expf(-((timestep_grouping - post_spike_g)*timestep) / post_decay): 0.0f;
          temp_post_trace *= (1.0f / (expf(-(timestep_grouping - pre_spike_g)*timestep / post_decay))); 
          syn_update_val -= stdp_vars.lambda * stdp_vars.alpha * old_synaptic_weight * temp_post_trace;
        }
        // OnPost Weight Update
        if (post_spike_g >= 0){
          float temp_pre_trace = stdp_pre_memory_trace_val;
          temp_pre_trace += (pre_spike_g > post_spike_g) ? -stdp_vars.a_plus*expf(-((timestep_grouping - pre_spike_g)*timestep) / pre_decay): 0.0f;
          temp_pre_trace *= (1.0f / (expf(-(timestep_grouping - post_spike_g)*timestep / pre_decay))); 
          syn_update_val += stdp_vars.lambda * (stdp_vars.w_max - old_synaptic_weight) * temp_pre_trace;
        }

        new_synaptic_weight = old_synaptic_weight + syn_update_val;
        if (new_synaptic_weight < 0.0f)
          new_synaptic_weight = 0.0f;
        
        // Weight Update
        d_synaptic_efficacies_or_weights[idx] = new_synaptic_weight;

        // Correctly set the trace values
        stdp_pre_memory_trace[indx] = stdp_pre_memory_trace_val;
        stdp_post_memory_trace[indx] = stdp_post_memory_trace_val;

        indx += blockDim.x * gridDim.x;
      }

    }
    
  }
}
