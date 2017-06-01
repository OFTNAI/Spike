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
                              sizeof(float)*frontend()->syns->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(recent_postsynaptic_activities_D,
                              frontend()->recent_postsynaptic_activities_D,
                              sizeof(float)*frontend()->neurs->total_number_of_neurons,
                              cudaMemcpyHostToDevice));
    }

    void EvansSTDPPlasticity::allocate_device_pointers(){
      // RUN AFTER NETWORK HAS BEEN STARTED
      // (eg, see prepare_backend() call at end of
      //  FourLayerVisionSpikingModel::finalise_model)
      CudaSafeCall(cudaMalloc((void **)&recent_postsynaptic_activities_D, sizeof(float)*frontend()->neurs->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&recent_presynaptic_activities_C, sizeof(float)*frontend()->syns->total_number_of_synapses));

    }

    void EvansSTDPPlasticity::update_synaptic_efficacies_or_weights(float current_time_in_seconds) {
      update_synaptic_efficacies_or_weights_kernel<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
        (recent_presynaptic_activities_C,
         recent_postsynaptic_activities_D,
         synapses_backend->postsynaptic_neuron_indices,
         synapses_backend->synaptic_efficacies_or_weights,
         current_time_in_seconds,
         synapses_backend->time_of_last_spike_to_reach_synapse,
         neurons_backend->last_spike_time_of_each_neuron,
         frontend()->stdp_params->learning_rate_rho,
         plastic_synapse_indices,
         total_number_of_plastic_synapses); // Here learning_rate_rho represents timestep/tau_delta_g in finite difference equation

      CudaCheckError();
    }

    void EvansSTDPPlasticity::update_presynaptic_activities(float timestep, float current_time_in_seconds) {
      update_presynaptic_activities_C_kernel<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>
        (recent_presynaptic_activities_C,
         synapses_backend->time_of_last_spike_to_reach_synapse,
         timestep,
         current_time_in_seconds,
         frontend()->stdp_params->synaptic_neurotransmitter_concentration_alpha_C,
         frontend()->stdp_params->decay_term_tau_C,
         plastic_synapse_indices,
         total_number_of_plastic_synapses);

      CudaCheckError();
    }

    void EvansSTDPPlasticity::update_postsynaptic_activities(float timestep, float current_time_in_seconds) {
      update_postsynaptic_activities_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
        (timestep,
         frontend()->neurs->total_number_of_neurons,
         recent_postsynaptic_activities_D,
         neurons_backend->last_spike_time_of_each_neuron,
         current_time_in_seconds,
         frontend()->stdp_params->decay_term_tau_D,
         frontend()->stdp_params->model_parameter_alpha_D);

	CudaCheckError();
    }

    __global__ void update_postsynaptic_activities_kernel
    (float timestep,
     size_t total_number_of_neurons,
     float * d_recent_postsynaptic_activities_D,
     float * d_last_spike_time_of_each_neuron,
     float current_time_in_seconds,
     float decay_term_tau_D,
     float model_parameter_alpha_D) {

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {

        // if (d_stdp[idx] == 1) {

        float recent_postsynaptic_activity_D = d_recent_postsynaptic_activities_D[idx];

        float new_recent_postsynaptic_activity_D = (1 - (timestep/decay_term_tau_D)) * recent_postsynaptic_activity_D;

        if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
          new_recent_postsynaptic_activity_D += timestep * model_parameter_alpha_D * (1 - recent_postsynaptic_activity_D);
        }
			
        d_recent_postsynaptic_activities_D[idx] = new_recent_postsynaptic_activity_D;

        // }

        idx += blockDim.x * gridDim.x;

      }
    }

    __global__ void update_presynaptic_activities_C_kernel
    (float* d_recent_presynaptic_activities_C,
     float* d_time_of_last_spike_to_reach_synapse,
     float timestep,
     float current_time_in_seconds,
     float synaptic_neurotransmitter_concentration_alpha_C,
     float decay_term_tau_C,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses) {

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < total_number_of_plastic_synapses) {
        int idx = d_plastic_synapse_indices[indx];

        float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];

        float new_recent_presynaptic_activity_C = (1 - (timestep/decay_term_tau_C)) * recent_presynaptic_activity_C;

        if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
          new_recent_presynaptic_activity_C += timestep * synaptic_neurotransmitter_concentration_alpha_C * (1 - recent_presynaptic_activity_C);
        }

        if (recent_presynaptic_activity_C != new_recent_presynaptic_activity_C) {
          d_recent_presynaptic_activities_C[idx] = new_recent_presynaptic_activity_C;
        }

        indx += blockDim.x * gridDim.x;

      }

    }

    __global__ void update_synaptic_efficacies_or_weights_kernel
    (float * d_recent_presynaptic_activities_C,
     float * d_recent_postsynaptic_activities_D,
     int* d_postsynaptic_neuron_indices,
     float* d_synaptic_efficacies_or_weights,
     float current_time_in_seconds,
     float * d_time_of_last_spike_to_reach_synapse,
     float * d_last_spike_time_of_each_neuron,
     float learning_rate_rho,
     int* d_plastic_synapse_indices,
     size_t total_number_of_plastic_synapses) {

      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      while (indx < total_number_of_plastic_synapses) {
        int idx = d_plastic_synapse_indices[indx];

        float synaptic_efficacy_delta_g = d_synaptic_efficacies_or_weights[idx];
        float new_synaptic_efficacy = synaptic_efficacy_delta_g;

        float new_componet = 0.0;

        int postsynaptic_neuron_index = d_postsynaptic_neuron_indices[idx];

        if (d_last_spike_time_of_each_neuron[postsynaptic_neuron_index] == current_time_in_seconds) {
          float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];
          float new_componet_addition = ((1 - synaptic_efficacy_delta_g) * recent_presynaptic_activity_C);
          new_componet += new_componet_addition;
        }

        if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
          float recent_postsynaptic_activity_D = d_recent_postsynaptic_activities_D[postsynaptic_neuron_index];
          new_componet -= (synaptic_efficacy_delta_g * recent_postsynaptic_activity_D);
        }			

        if (new_componet != 0.0) {
          new_componet = learning_rate_rho * new_componet;
          new_synaptic_efficacy += new_componet;
        }
		
        if (synaptic_efficacy_delta_g != new_synaptic_efficacy) {
          new_synaptic_efficacy = max(new_synaptic_efficacy, 0.0);
          new_synaptic_efficacy = min(new_synaptic_efficacy, 1.0);

          d_synaptic_efficacies_or_weights[idx] = new_synaptic_efficacy;
        }

 

        indx += blockDim.x * gridDim.x;
      }
    }

  }
}
