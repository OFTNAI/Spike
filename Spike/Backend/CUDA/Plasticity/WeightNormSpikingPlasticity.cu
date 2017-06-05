// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/WeightNormSpikingPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, WeightNormSpikingPlasticity);

namespace Backend {
  namespace CUDA {
    WeightNormSpikingPlasticity::~WeightNormSpikingPlasticity() {    
        CudaSafeCall(cudaFree(plastic_synapse_indices));
        CudaSafeCall(cudaFree(sum_squared_afferent_values));
        CudaSafeCall(cudaFree(neuron_in_plasticity_set));
        CudaSafeCall(cudaFree(initial_weights));
        CudaSafeCall(cudaFree(weight_divisor));
    }

    void WeightNormSpikingPlasticity::reset_state() {
	if (total_number_of_plastic_synapses > 0) {
		float* weight_values = (float*)malloc(sizeof(float)*total_number_of_plastic_synapses);
		for (int id = 0; id < total_number_of_plastic_synapses; id++){
			weight_values[id] = frontend()->syns->synaptic_efficacies_or_weights[
				frontend()->syns->plasticity_synapse_indices_per_rule[frontend()->plasticity_rule_id][id]];
		}

		// Now load values into device memory
		CudaSafeCall(cudaMemcpy((void*)initial_weights, weight_values, sizeof(float)*total_number_of_plastic_synapses, cudaMemcpyHostToDevice));
	  	free(weight_values);

		CudaSafeCall(cudaMemcpy((void*)afferent_weight_change_updater,
				(void*)frontend()->afferent_weight_change_updater,
				sizeof(float)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));
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
	  CudaSafeCall(cudaMalloc((void **)&sum_squared_afferent_values, sizeof(float)*frontend()->neurs->total_number_of_neurons));
	  CudaSafeCall(cudaMalloc((void **)&afferent_weight_change_updater, sizeof(float)*frontend()->neurs->total_number_of_neurons));
	  CudaSafeCall(cudaMalloc((void **)&neuron_in_plasticity_set, sizeof(bool)*frontend()->neurs->total_number_of_neurons));
	  // Copy values
	  CudaSafeCall(cudaMemcpy((void*)sum_squared_afferent_values,
				(void*)frontend()->sum_squared_afferent_values,
				sizeof(float)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));
	  CudaSafeCall(cudaMemcpy((void*)afferent_weight_change_updater,
				(void*)frontend()->afferent_weight_change_updater,
				sizeof(float)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));
	  CudaSafeCall(cudaMemcpy((void*)neuron_in_plasticity_set,
				(void*)frontend()->neuron_in_plasticity_set,
				sizeof(bool)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));

	  // Loading initial weights and setting weight changes to zero
	  CudaSafeCall(cudaMalloc((void **)&initial_weights, sizeof(float)*total_number_of_plastic_synapses));
	  CudaSafeCall(cudaMalloc((void **)&weight_divisor, sizeof(float)*frontend()->neurs->total_number_of_neurons));
	}
    }

    void WeightNormSpikingPlasticity::weight_normalization(){
	if (total_number_of_plastic_synapses > 0) {
	CudaSafeCall(cudaMemcpy((void*)afferent_weight_change_updater,
			(void*)frontend()->afferent_weight_change_updater,
			sizeof(float)*frontend()->neurs->total_number_of_neurons, cudaMemcpyHostToDevice));

	// First calculate the weight change
	weight_change_calculations<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>(
		synapses_backend->postsynaptic_neuron_indices,
		synapses_backend->synaptic_efficacies_or_weights,
		initial_weights,
		afferent_weight_change_updater,
		plastic_synapse_indices,
		total_number_of_plastic_synapses);
	CudaCheckError();
	weight_division_calc<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>(
		sum_squared_afferent_values,
		afferent_weight_change_updater,
		weight_divisor,
		neuron_in_plasticity_set,
		frontend()->neurs->total_number_of_neurons);
	CudaCheckError();
	weight_update<<<synapses_backend->number_of_synapse_blocks_per_grid, synapses_backend->threads_per_block>>>(
		synapses_backend->postsynaptic_neuron_indices,
		neuron_in_plasticity_set,
		synapses_backend->synaptic_efficacies_or_weights,
		weight_divisor,
		plastic_synapse_indices,
		total_number_of_plastic_synapses);
	CudaCheckError();
	}	
    }

    __global__ void weight_change_calculations(
		int* postsyn_ids,
		float* current_weight,
		float* initial_weights,
		float* afferent_weight_change_updater,
		int* d_plastic_synapse_indices,
		size_t total_number_of_plastic_synapses)
	{
		// Global Index
		int indx = threadIdx.x + blockIdx.x * blockDim.x;
		
		while (indx < total_number_of_plastic_synapses) {
			// Get the current synapse index
			int idx = d_plastic_synapse_indices[indx];
			int post_id = postsyn_ids[idx];
			float weight_change = current_weight[idx] - initial_weights[indx];
			if (weight_change != 0.0){
				float update_value = weight_change*weight_change + 2.0f*initial_weights[indx]*weight_change;
				atomicAdd(&afferent_weight_change_updater[post_id], update_value);
			}
			indx += blockDim.x * gridDim.x;
		}
		__syncthreads();
	}

	  __global__ void weight_division_calc(
		float* sum_squared_afferent_values,
		float* afferent_weight_change_updater,
		float* weight_divisor,
		bool* neuron_in_plasticity_set,
		size_t total_number_of_neurons)
	{
		// Global Index
		int idx = threadIdx.x + blockIdx.x * blockDim.x;

		while (idx < total_number_of_neurons) {
			if (neuron_in_plasticity_set[idx])
			{
				if ((sum_squared_afferent_values[idx] - afferent_weight_change_updater[idx] < 0.01))
					printf("NORMALIZATION DIFF VERY LARGE. DANGER OF SYNAPSES ALL -> ZERO");
				weight_divisor[idx] = sqrtf(sum_squared_afferent_values[idx] + afferent_weight_change_updater[idx]) / sqrtf(sum_squared_afferent_values[idx]);
			}
			idx += blockDim.x * gridDim.x;		
		}
		__syncthreads();
	}


	__global__ void weight_update(
		int* postsyn_neuron,
		bool* neuron_in_plasticity_set,
		float* current_weight,
		float* weight_divisor,
		int* d_plastic_synapse_indices,
		size_t total_number_of_plastic_synapses){
	
		// Global Index
		int indx = threadIdx.x + blockIdx.x * blockDim.x;
		
		while (indx < total_number_of_plastic_synapses) {
			int idx = d_plastic_synapse_indices[indx];
			int postneuron = postsyn_neuron[idx];
			if (neuron_in_plasticity_set[postneuron]){
				float division_value = weight_divisor[postneuron];
				//if (division_value != 1.0)
				//printf("%f, %f, %f wat \n", division_value, current_weight[idx], (current_weight[idx] / division_value));
				if (division_value != 1.0)
					current_weight[idx] /= division_value;
			}
			indx += blockDim.x * gridDim.x;
		}
	}


  }
}
