// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"

namespace Backend {
  namespace CUDA {
    // ConductanceSpikingSynapses Destructor
    ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
      CudaSafeCall(cudaFree(synaptic_conductances_g));
      CudaSafeCall(cudaFree(biological_conductance_scaling_constants_lambda));
      CudaSafeCall(cudaFree(reversal_potentials_Vhat));
      CudaSafeCall(cudaFree(decay_terms_tau_g));
    }

    void ConductanceSpikingSynapses::calculate_postsynaptic_current_injection(::SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {

      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons->backend());
      assert(neurons_backend);

      	conductance_calculate_postsynaptic_current_injection_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>
          (presynaptic_neuron_indices,
           postsynaptic_neuron_indices,
           reversal_potentials_Vhat,
           neurons_backend->current_injections,
           frontend()->total_number_of_synapses,
           neurons_backend->membrane_potentials_v, 
           synaptic_conductances_g);

	CudaCheckError();
    }

    void ConductanceSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
      CudaSafeCall(cudaMemcpy(synaptic_conductances_g,
                              frontend()->synaptic_conductances_g,
                              sizeof(float)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
    }

    void ConductanceSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
    }

    void ConductanceSpikingSynapses::push_data_front() {
      SpikingSynapses::push_data_front();
    }

    void ConductanceSpikingSynapses::pull_data_back() {
      SpikingSynapses::pull_data_back();
    }

    void ConductanceSpikingSynapses::allocate_device_pointers() {
	CudaSafeCall(cudaMalloc((void **)&biological_conductance_scaling_constants_lambda, sizeof(float)*frontend()->total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&reversal_potentials_Vhat, sizeof(float)*frontend()->total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&decay_terms_tau_g, sizeof(float)*frontend()->total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&synaptic_conductances_g, sizeof(float)*frontend()->total_number_of_synapses));
    }

    void ConductanceSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(biological_conductance_scaling_constants_lambda, frontend()->biological_conductance_scaling_constants_lambda, sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(reversal_potentials_Vhat, frontend()->reversal_potentials_Vhat, sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(decay_terms_tau_g, frontend()->decay_terms_tau_g, sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
    }

    void ConductanceSpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {
	conductance_update_synaptic_conductances_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>
          (timestep, 
           synaptic_conductances_g, 
           synaptic_efficacies_or_weights, 
           time_of_last_spike_to_reach_synapse,
           biological_conductance_scaling_constants_lambda,
           frontend()->total_number_of_synapses,
           current_time_in_seconds,
           decay_terms_tau_g);

	CudaCheckError();
    }


    __global__ void conductance_calculate_postsynaptic_current_injection_kernel
    (int * d_presynaptic_neuron_indices,
     int* d_postsynaptic_neuron_indices,
     float* d_reversal_potentials_Vhat,
     float* d_neurons_current_injections,
     size_t total_number_of_synapses,
     float * d_membrane_potentials_v,
     float * d_synaptic_conductances_g){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_synapses) {

        float reversal_potential_Vhat = d_reversal_potentials_Vhat[idx];
        int postsynaptic_neuron_index = d_postsynaptic_neuron_indices[idx];
        float membrane_potential_v = d_membrane_potentials_v[postsynaptic_neuron_index];
        float synaptic_conductance_g = d_synaptic_conductances_g[idx];

        float component_for_sum = synaptic_conductance_g * (reversal_potential_Vhat - membrane_potential_v);
        if (component_for_sum != 0.0) {
          atomicAdd(&d_neurons_current_injections[postsynaptic_neuron_index], component_for_sum);
        }

        idx += blockDim.x * gridDim.x;

      }
      __syncthreads();
    }

    __global__ void conductance_update_synaptic_conductances_kernel
    (float timestep, 
     float * d_synaptic_conductances_g, 
     float * d_synaptic_efficacies_or_weights, 
     float * d_time_of_last_spike_to_reach_synapse,
     float * d_biological_conductance_scaling_constants_lambda,
     int total_number_of_synapses,
     float current_time_in_seconds,
     float * d_decay_terms_tau_g) {

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_synapses) {

        float synaptic_conductance_g = d_synaptic_conductances_g[idx];
        float new_conductance = (1.0 - (timestep/d_decay_terms_tau_g[idx])) * synaptic_conductance_g;
		
        if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
          float timestep_times_synaptic_efficacy = timestep * d_synaptic_efficacies_or_weights[idx];
          float biological_conductance_scaling_constant_lambda = d_biological_conductance_scaling_constants_lambda[idx];
          float timestep_times_synaptic_efficacy_times_scaling_constant = timestep_times_synaptic_efficacy * biological_conductance_scaling_constant_lambda;
          new_conductance += timestep_times_synaptic_efficacy_times_scaling_constant;
        }

        if (synaptic_conductance_g != new_conductance) {
          d_synaptic_conductances_g[idx] = new_conductance;
        }

        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }
  }
}
