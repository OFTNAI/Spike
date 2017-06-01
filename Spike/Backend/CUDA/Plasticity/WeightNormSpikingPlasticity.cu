// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/WeightNormSpikingPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, WeightNormSpikingPlasticity);

namespace Backend {
  namespace CUDA {
    WeightNormSpikingPlasticity::~WeightNormSpikingPlasticity() {    
    }

    void WeightNormSpikingPlasticity::reset_state() {
	if (total_number_of_plastic_synapses > 0) {
		float* weight_values = (float*)malloc(sizeof(float)*total_number_of_plastic_synapses);
		float* weight_change_init = (float*)malloc(sizeof(float)*total_number_of_plastic_synapses);

		for (int id = 0; id < total_number_of_plastic_synapses; id++){
			weight_change_init[id] = 0.0f;
			weight_values[id] = frontend()->syns->synaptic_efficacies_or_weights[
				frontend()->syns->plasticity_synapse_indices_per_rule[frontend()->plasticity_rule_id][id]];
		}

		// Now load values into device memory
		CudaSafeCall(cudaMemcpy((void*)initial_weights, weight_values, sizeof(float)*total_number_of_plastic_synapses, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy((void*)weight_changes, weight_change_init, sizeof(float)*total_number_of_plastic_synapses, cudaMemcpyHostToDevice));
    	}
    }

    void WeightNormSpikingPlasticity::prepare() {
      
      // Set up synapses backend and synaptic details
      synapses_backend = dynamic_cast<::Backend::CUDA::Synapses*>
	(frontend()->syns->backend());
      int plasticity_id = frontend()->plasticity_rule_id;
      if (plasticity_id >= 0) {
	total_number_of_plastic_synapses = frontend()->syns->plasticity_synapse_number_per_rule[plasticity_id];
      } else {
	total_number_of_plastic_synapses = 0;
      }

      // This learning rule requires a device side storage of a number of variables

      allocate_device_pointers();
    }

    void WeightNormSpikingPlasticity::allocate_device_pointers() {
	if (total_number_of_plastic_synapses > 0){
	  CudaSafeCall(cudaMalloc((void **)&plastic_synapse_indices, sizeof(int)*total_number_of_plastic_synapses));
	  CudaSafeCall(cudaMemcpy((void*)plastic_synapse_indices,
				  (void*)frontend()->syns->plasticity_synapse_indices_per_rule[frontend()->plasticity_rule_id],
				  sizeof(int)*total_number_of_plastic_synapses,
				  cudaMemcpyHostToDevice));
	  // Loading vectors from front-end
	  CudaSafeCall(cudaMalloc((void **)&total_afferent_synapse_initial, sizeof(float)*frontend()->neurs->total_number_of_neurons));
	  CudaSafeCall(cudaMalloc((void **)&afferent_synapse_changes, sizeof(float)*frontend()->neurs->total_number_of_neurons));
	  CudaSafeCall(cudaMalloc((void **)&neuron_in_plasticity_set, sizeof(bool)*frontend()->neurs->total_number_of_neurons));
	  // Copy values
	  CudaSafeCall(cudaMemcpy((void*)total_afferent_synapse_initial,
				(void*)frontend()->total_afferent_synapse_initial,
				sizeof(float)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));
	  CudaSafeCall(cudaMemcpy((void*)afferent_synapse_changes,
				(void*)frontend()->afferent_synapse_changes,
				sizeof(float)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));
	  CudaSafeCall(cudaMemcpy((void*)neuron_in_plasticity_set,
				(void*)frontend()->neuron_in_plasticity_set,
				sizeof(bool)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));

	  // Loading initial weights and setting weight changes to zero
	  CudaSafeCall(cudaMalloc((void **)&initial_weights, sizeof(float)*total_number_of_plastic_synapses));
	  CudaSafeCall(cudaMalloc((void **)&weight_changes, sizeof(float)*total_number_of_plastic_synapses));
	}
    }

    void WeightNormSpikingPlasticity::weight_normalization(){
	// First calculate the weight change
	weight_change_calculations<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>(
		synapses_backend->synaptic_efficacies_or_weights,
		initial_weights,
		weight_changes,
		plastic_synapse_indices,
		total_number_of_plastic_synapses);	
    }

    __global__ void weight_change_calculations(
		float* current_weight,
		float* initial_weights,
		float* weight_changes,
		int* d_plastic_synapse_indices,
		size_t total_number_of_plastic_synapses)
	{
		// Global Index
		int indx = threadIdx.x + blockIdx.x * blockDim.x;
		
		while (indx < total_number_of_plastic_synapses) {
			// Get the current synapse index
			int idx = d_plastic_synapse_indices[indx];
			
			weight_changes[indx] = initial_weights[indx] + current_weight[idx];

			indx += blockDim.x * globalDim.x;
		}
	}
  }
}
