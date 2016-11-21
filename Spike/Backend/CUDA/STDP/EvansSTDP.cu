#include "Spike/Backend/CUDA/STDP/EvansSTDP.hpp"

namespace Backend {
  namespace CUDA {
    EvansSTDP::~EvansSTDP() {
      // TODO Check before free
      CudaSafeCall(cudaFree(d_recent_postsynaptic_activities_D));
      CudaSafeCall(cudaFree(d_recent_presynaptic_activities_C));
    }

    void EvansSTDP::reset_state() {
      CudaSafeCall(cudaMemcpy(d_recent_presynaptic_activities_C, recent_presynaptic_activities_C, sizeof(float)*syns->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(d_recent_postsynaptic_activities_D, recent_postsynaptic_activities_D, sizeof(float)*neurs->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    // RUN AFTER NETWORK HAS BEEN STARTED
    void EvansSTDP::allocate_device_pointers(){
      // Create extra LIF arrays
      recent_postsynaptic_activities_D = (float*)realloc(recent_postsynaptic_activities_D, (neurs->total_number_of_neurons*sizeof(float)));
      recent_presynaptic_activities_C = (float*)realloc(recent_presynaptic_activities_C, syns->total_number_of_synapses*sizeof(float));
      for (int i = 0; i < neurs->total_number_of_neurons; i++){
        recent_postsynaptic_activities_D[i] = 0.0f;
      }
      for (int i = 0; i < syns->total_number_of_synapses; i++){
        recent_presynaptic_activities_C[i] = 0.0f;
      }

      // CUDA them
      //
      CudaSafeCall(cudaMalloc((void **)&d_recent_postsynaptic_activities_D, sizeof(float)*neurs->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_recent_presynaptic_activities_C, sizeof(float)*syns->total_number_of_synapses));

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
     bool* d_stdp,
     float timestep, 
     float current_time_in_seconds,
     size_t total_number_of_synapses,
     float synaptic_neurotransmitter_concentration_alpha_C,
     float decay_term_tau_C) {

      int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
      int idx = t_idx;
      while (idx < total_number_of_synapses) {

        if (d_stdp[idx] == true) {

          float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];

          float new_recent_presynaptic_activity_C = (1 - (timestep/decay_term_tau_C)) * recent_presynaptic_activity_C;

          if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
            new_recent_presynaptic_activity_C += timestep * synaptic_neurotransmitter_concentration_alpha_C * (1 - recent_presynaptic_activity_C);
          }

          if (recent_presynaptic_activity_C != new_recent_presynaptic_activity_C) {
            d_recent_presynaptic_activities_C[idx] = new_recent_presynaptic_activity_C;
          }

        }

        idx += blockDim.x * gridDim.x;

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
     bool* d_stdp,
     size_t total_number_of_synapses,
     float learning_rate_rho) {

      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      while (idx < total_number_of_synapses) {

        if (d_stdp[idx] == true) {

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

        }

        idx += blockDim.x * gridDim.x;
      }
    }

  }
}
