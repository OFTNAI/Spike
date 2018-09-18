// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/EvansSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, EvansSTDPPlasticity);

namespace Backend {
  namespace CUDA {
    EvansSTDPPlasticity::~EvansSTDPPlasticity() {
      CudaSafeCall(cudaFree(recent_postsynaptic_activities_D));
      CudaSafeCall(cudaFree(recent_presynaptic_activities_C));
    }

    void EvansSTDPPlasticity::prepare() {
      STDPPlasticity::prepare();

      allocate_device_pointers();
    }

    void EvansSTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();


      CudaSafeCall(cudaMemcpy(recent_presynaptic_activities_C,
                              frontend()->recent_presynaptic_activities_C,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(recent_postsynaptic_activities_D,
                              frontend()->recent_postsynaptic_activities_D,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
    }

    void EvansSTDPPlasticity::allocate_device_pointers(){
      // RUN AFTER NETWORK HAS BEEN STARTED
      // (eg, see prepare_backend() call at end of
      //  FourLayerVisionSpikingModel::finalise_model)
      CudaSafeCall(cudaMalloc((void **)&recent_postsynaptic_activities_D, sizeof(float)*frontend()->neurs->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&recent_presynaptic_activities_C, sizeof(float)*frontend()->syns->total_number_of_synapses));

    }

    void EvansSTDPPlasticity::update_synaptic_efficacies_or_weights(float current_time_in_seconds, float timestep) {
        ltp_and_ltd<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
          (synapses_backend->postsynaptic_neuron_indices,
           synapses_backend->presynaptic_neuron_indices,
           synapses_backend->delays,
           neurons_backend->d_neuron_data,
           input_neurons_backend->d_neuron_data,
           synapses_backend->synaptic_efficacies_or_weights,
           recent_presynaptic_activities_C,
           recent_postsynaptic_activities_D,
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
           float* recent_presynaptic_activities_C,
           float* recent_postsynaptic_activities_D,
           evans_stdp_plasticity_parameters_struct stdp_vars,
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
        
        // Getting synapse details
        float recent_presynaptic_activity_C = recent_presynaptic_activities_C[indx];
        float recent_postsynaptic_activity_D = recent_postsynaptic_activities_D[indx];
        int postid = d_postsyns[idx];
        int preid = d_presyns[idx];
        int bufsize = input_neuron_data->neuron_spike_time_bitbuffer_bytesize[0];
        float old_synaptic_weight = d_synaptic_efficacies_or_weights[idx];
        float new_synaptic_weight = old_synaptic_weight;

        // Correcting for input vs output neuron types
        bool is_input = PRESYNAPTIC_IS_INPUT(preid);
        int corr_preid = CORRECTED_PRESYNAPTIC_ID(preid, is_input);
        uint8_t* pre_bitbuffer = is_input ? input_neuron_data->neuron_spike_time_bitbuffer : neuron_data->neuron_spike_time_bitbuffer;

        // Looping over timesteps
        for (int g=0; g < timestep_grouping; g++){
          // Decaying STDP traces
          recent_presynaptic_activity_C = (1 - (timestep/stdp_vars.decay_term_tau_C)) * recent_presynaptic_activity_C;
          recent_postsynaptic_activity_D = (1 - (timestep/stdp_vars.decay_term_tau_D)) * recent_postsynaptic_activity_D;

          // Bit Indexing to detect spikes
          int postbitloc = ((int)roundf(current_time_in_seconds / timestep) + g) % (bufsize*8);
          int prebitloc = postbitloc - d_syndelays[idx];
          prebitloc = (prebitloc < 0) ? (bufsize*8 + prebitloc) : prebitloc;

          // OnPre Trace Update
          if (pre_bitbuffer[corr_preid*bufsize + (prebitloc / 8)] & (1 << (prebitloc % 8))){
            recent_presynaptic_activity_C += timestep * stdp_vars.synaptic_neurotransmitter_concentration_alpha_C * (1 - recent_presynaptic_activity_C);
          }
          // OnPost Trace Update
          if (neuron_data->neuron_spike_time_bitbuffer[postid*bufsize + (postbitloc / 8)] & (1 << (postbitloc % 8))){
            recent_postsynaptic_activity_D += timestep * stdp_vars.model_parameter_alpha_D * (1 - recent_postsynaptic_activity_D);
          }
          
          float syn_update_val = 0.0f; 
          old_synaptic_weight = new_synaptic_weight;
          // OnPre Weight Update
          if (pre_bitbuffer[corr_preid*bufsize + (prebitloc / 8)] & (1 << (prebitloc % 8))){
            syn_update_val -= (old_synaptic_weight * recent_postsynaptic_activity_D);
          }
          // OnPost Weight Update
          if (neuron_data->neuron_spike_time_bitbuffer[postid*bufsize + (postbitloc / 8)] & (1 << (postbitloc % 8))){
            syn_update_val += ((1 - old_synaptic_weight) * recent_presynaptic_activity_C);
          }

          new_synaptic_weight = old_synaptic_weight + syn_update_val;
          if (new_synaptic_weight < 0.0f)
            new_synaptic_weight = 0.0f;
          if (new_synaptic_weight > 1.0f)
            new_synaptic_weight = 1.0f;
        }
        
        // Weight Update
        d_synaptic_efficacies_or_weights[idx] = new_synaptic_weight;

        // Correctly set the trace values
        recent_presynaptic_activities_C[indx] = recent_presynaptic_activity_C;
        recent_postsynaptic_activities_D[indx] = recent_postsynaptic_activity_D;

        indx += blockDim.x * gridDim.x;
      }

    }


  }
}
