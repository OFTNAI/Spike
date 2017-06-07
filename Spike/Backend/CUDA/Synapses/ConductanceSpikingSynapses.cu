// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, ConductanceSpikingSynapses);

namespace Backend {
  namespace CUDA {
    // ConductanceSpikingSynapses Destructor
    ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
      CudaSafeCall(cudaFree(synaptic_conductances_g));
      CudaSafeCall(cudaFree(biological_conductance_scaling_constants_lambda));
      CudaSafeCall(cudaFree(reversal_potentials_Vhat));
      CudaSafeCall(cudaFree(decay_terms_tau_g));
      CudaSafeCall(cudaFree(num_active_synapses));
      CudaSafeCall(cudaFree(active_synapse_indices));
#ifdef CRAZY_DEBUG
      std::cout << "\n!!!!!!!!!!!!!!!!!!!!---CCCCCC---!!!!!!!!!!!!!!!!!!!\n";
#endif
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
           num_active_synapses,
           active_synapse_indices,
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
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
    }

    void ConductanceSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
    }

    void ConductanceSpikingSynapses::allocate_device_pointers() {
	CudaSafeCall(cudaMalloc((void **)&biological_conductance_scaling_constants_lambda, sizeof(float)*frontend()->total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&reversal_potentials_Vhat, sizeof(float)*frontend()->total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&decay_terms_tau_g, sizeof(float)*frontend()->total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&synaptic_conductances_g, sizeof(float)*frontend()->total_number_of_synapses));
  CudaSafeCall(cudaMalloc((void **)&active_synapse_indices, sizeof(int)*frontend()->total_number_of_synapses));
  CudaSafeCall(cudaMalloc((void **)&num_active_synapses, sizeof(int)));
    }

    void ConductanceSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(biological_conductance_scaling_constants_lambda, frontend()->biological_conductance_scaling_constants_lambda, sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(reversal_potentials_Vhat, frontend()->reversal_potentials_Vhat, sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(decay_terms_tau_g, frontend()->decay_terms_tau_g, sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
    }

    void ConductanceSpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {
	conductance_update_synaptic_conductances_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>
          (timestep, 
           synaptic_conductances_g, 
           synaptic_efficacies_or_weights, 
           time_of_last_spike_to_reach_synapse,
           biological_conductance_scaling_constants_lambda,
           num_active_synapses,
           active_synapse_indices,
           current_time_in_seconds,
           decay_terms_tau_g);

	CudaCheckError();
    }

    void ConductanceSpikingSynapses::interact_spikes_with_synapses
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {

      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons->backend());
      assert(neurons_backend);
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(input_neurons->backend());
      assert(input_neurons_backend);

      // Reset active Synapses
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      // Get Active Synapses
      get_active_synapses_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(presynaptic_neuron_indices,
                      delays,
                      spikes_travelling_to_synapse,
		      neurons_backend->last_spike_time_of_each_neuron,
                      input_neurons_backend->last_spike_time_of_each_neuron,
                      synaptic_conductances_g,
                      decay_terms_tau_g,
                      current_time_in_seconds,
                      num_active_synapses,
                      active_synapse_indices,
                      timestep,
                      frontend()->total_number_of_synapses);
      CudaCheckError();

      if (neurons_backend->frontend()->high_fidelity_spike_flag){
      conductance_check_bitarray_for_presynaptic_neuron_spikes<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(
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
    }
    else{
      conductance_move_spikes_towards_synapses_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(
                                        presynaptic_neuron_indices,
                                        delays,
                                        spikes_travelling_to_synapse,
                                        neurons_backend->last_spike_time_of_each_neuron,
                                        input_neurons_backend->last_spike_time_of_each_neuron,
                                        current_time_in_seconds,
                                        num_active_synapses,
                                        active_synapse_indices,
                                        time_of_last_spike_to_reach_synapse);
      CudaCheckError();
    }

    }


    __global__ void get_active_synapses_kernel(int* d_presynaptic_neuron_indices,
                    int* d_delays,
		    int* d_spikes_travelling_to_synapse, 
                    float* d_last_spike_time_of_each_neuron,
                    float* d_input_neurons_last_spike_time,
                    float * d_synaptic_conductances_g,
                    float * d_decay_terms_tau_g,
                    float current_time_in_seconds,
                    int* d_num_active_synapses,
                    int* d_active_synapses,
                    float timestep,
                    size_t total_number_of_synapses){

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_synapses) {

        int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
        bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
        int delay = d_delays[idx];
        float effecttime = presynaptic_is_input ? d_input_neurons_last_spike_time[CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input)] : d_last_spike_time_of_each_neuron[presynaptic_neuron_index];

	if ((effecttime == current_time_in_seconds) && (d_spikes_travelling_to_synapse[idx] == 0))
		d_spikes_travelling_to_synapse[idx] = -1;

        // Add to the effect time the length of a delay and a decay time ~10 tau (When current injection has reduced to 0.005% of the original value)
        effecttime += (delay + 1)*timestep + 10.0f*d_decay_terms_tau_g[idx];

        if (effecttime > current_time_in_seconds){
          int pos = atomicAdd(&d_num_active_synapses[0], 1);
          d_active_synapses[pos] = idx;
        }

      	__syncthreads();
        idx += blockDim.x * gridDim.x;
      }
    }

    __global__ void conductance_calculate_postsynaptic_current_injection_kernel(int * d_presynaptic_neuron_indices,
                  int* d_postsynaptic_neuron_indices,
                  float* d_reversal_potentials_Vhat,
                  float* d_neurons_current_injections,
                  int* d_num_active_synapses,
                  int* d_active_synapses,
                  float * d_membrane_potentials_v,
                  float * d_synaptic_conductances_g){

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < d_num_active_synapses[0]) {
        int idx = d_active_synapses[indx];

        float reversal_potential_Vhat = d_reversal_potentials_Vhat[idx];
        int postsynaptic_neuron_index = d_postsynaptic_neuron_indices[idx];
        float membrane_potential_v = d_membrane_potentials_v[postsynaptic_neuron_index];
        float synaptic_conductance_g = d_synaptic_conductances_g[idx];

        float component_for_sum = synaptic_conductance_g * (reversal_potential_Vhat - membrane_potential_v);
        if (component_for_sum != 0.0) {
          atomicAdd(&d_neurons_current_injections[postsynaptic_neuron_index], component_for_sum);
        }

      	__syncthreads();
        indx += blockDim.x * gridDim.x;

      }
    }

    __global__ void conductance_update_synaptic_conductances_kernel(float timestep,
                                float * d_synaptic_conductances_g,
                                float * d_synaptic_efficacies_or_weights,
                                float * d_time_of_last_spike_to_reach_synapse,
                                float * d_biological_conductance_scaling_constants_lambda,
                                int* d_num_active_synapses,
                                int* d_active_synapses,
                                float current_time_in_seconds,
                                float * d_decay_terms_tau_g) {

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < d_num_active_synapses[0]) {
        int idx = d_active_synapses[indx];

        float synaptic_conductance_g = d_synaptic_conductances_g[idx];

        float new_conductance = (1.0 - (timestep/d_decay_terms_tau_g[idx])) * synaptic_conductance_g;

        if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
          float synaptic_efficacy = d_synaptic_efficacies_or_weights[idx];
          float biological_conductance_scaling_constant_lambda = d_biological_conductance_scaling_constants_lambda[idx];
          float synaptic_efficacy_times_scaling_constant = synaptic_efficacy * biological_conductance_scaling_constant_lambda;
          new_conductance += synaptic_efficacy_times_scaling_constant;
        }

        d_synaptic_conductances_g[idx] = new_conductance;

        indx += blockDim.x * gridDim.x;
      }
      __syncthreads();

    }

    __global__ void conductance_move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
                    int* d_delays,
                    int* d_spikes_travelling_to_synapse,
                    float* d_last_spike_time_of_each_neuron,
                    float* d_input_neurons_last_spike_time,
                    float current_time_in_seconds,
                    int* d_num_active_synapses,
                    int* d_active_synapses,
                    float* d_time_of_last_spike_to_reach_synapse){

      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      while (indx < d_num_active_synapses[0]) {
        int idx = d_active_synapses[indx];

        int timesteps_until_spike_reaches_synapse = d_spikes_travelling_to_synapse[idx];

	if (timesteps_until_spike_reaches_synapse > 0) {
	  timesteps_until_spike_reaches_synapse -= 1;

   	  if (timesteps_until_spike_reaches_synapse == 0)
            d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
        }

        if (timesteps_until_spike_reaches_synapse < 0)
            timesteps_until_spike_reaches_synapse = d_delays[idx];
   

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
  }
}
