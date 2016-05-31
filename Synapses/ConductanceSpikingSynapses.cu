#include "ConductanceSpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

// ConductanceSpikingSynapses Constructor
ConductanceSpikingSynapses::ConductanceSpikingSynapses() {

	synaptic_conductances_g = NULL;
	d_synaptic_conductances_g = NULL;

	recent_presynaptic_activities_C = NULL;
	d_recent_presynaptic_activities_C = NULL;

	biological_conductance_scaling_constants_lambda = NULL;
	d_biological_conductance_scaling_constants_lambda = NULL;

	reversal_potentials_Vhat = NULL;
	d_reversal_potentials_Vhat = NULL;

}

// ConductanceSpikingSynapses Destructor
ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
	free(synaptic_conductances_g);
	CudaSafeCall(cudaFree(d_synaptic_conductances_g));

	free(recent_presynaptic_activities_C);
	CudaSafeCall(cudaFree(d_recent_presynaptic_activities_C));
}

// Connection Detail implementation
//	INPUT:
//		Pre-neuron population ID
//		Post-neuron population ID
//		An array of the exclusive sum of neuron populations
//		CONNECTIVITY_TYPE (Constants.h)
//		2 number float array for weight range
//		2 number float array for delay range
//		Boolean value to indicate if population is STDP based
//		Parameter = either probability for random synapses or S.D. for Gaussian
void ConductanceSpikingSynapses::AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int delay_range[2],
						synapse_parameters_struct * synapse_params,
						float parameter,
						float parameter_two) {
	
	
	SpikingSynapses::AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							delay_range,
							synapse_params,
							parameter,
							parameter_two);

	conductance_spiking_synapse_parameters_struct * conductance_spiking_synapse_group_params = (conductance_spiking_synapse_parameters_struct*)synapse_params;

	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses-1; i++){
		synaptic_conductances_g[i] = 0.0f;
		recent_presynaptic_activities_C[i] = 0.0f;
		biological_conductance_scaling_constants_lambda[i] = conductance_spiking_synapse_group_params->biological_conductance_scaling_constant_lambda;
		reversal_potentials_Vhat[i] = conductance_spiking_synapse_group_params->reversal_potential_Vhat;
	}

}

void ConductanceSpikingSynapses::increment_number_of_synapses(int increment) {

	SpikingSynapses::increment_number_of_synapses(increment);

	synaptic_conductances_g = (float*)realloc(synaptic_conductances_g, total_number_of_synapses * sizeof(float));
	recent_presynaptic_activities_C = (float*)realloc(recent_presynaptic_activities_C, total_number_of_synapses * sizeof(float));
	biological_conductance_scaling_constants_lambda = (float*)realloc(biological_conductance_scaling_constants_lambda, total_number_of_synapses * sizeof(float));
	reversal_potentials_Vhat = (float*)realloc(reversal_potentials_Vhat, total_number_of_synapses * sizeof(float));

}


void ConductanceSpikingSynapses::allocate_device_pointers() {

	SpikingSynapses::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_synaptic_conductances_g, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_recent_presynaptic_activities_C, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_biological_conductance_scaling_constants_lambda, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_reversal_potentials_Vhat, sizeof(float)*total_number_of_synapses));

}

void ConductanceSpikingSynapses::reset_synapse_spikes() {

	SpikingSynapses::reset_synapse_spikes();

	// CudaSafeCall(cudaMemset(d_synaptic_conductances_g, 0.0f, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMemcpy(d_synaptic_conductances_g, synaptic_conductances_g, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_recent_presynaptic_activities_C, recent_presynaptic_activities_C, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_biological_conductance_scaling_constants_lambda, biological_conductance_scaling_constants_lambda, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_reversal_potentials_Vhat, reversal_potentials_Vhat, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));

}


void ConductanceSpikingSynapses::shuffle_synapses() {
	SpikingSynapses::shuffle_synapses();
}


void ConductanceSpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	SpikingSynapses::set_threads_per_block_and_blocks_per_grid(threads);
	
}



void ConductanceSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds) {

	// printf("calculate_postsynaptic_current_injection. number_of_synapse_blocks_per_grid.x: %d. threads_per_block.x: %d\n", number_of_synapse_blocks_per_grid.x, threads_per_block.x);

	conductance_calculate_postsynaptic_current_injection_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																	d_postsynaptic_neuron_indices,
																	d_reversal_potentials_Vhat,
																	neurons->d_current_injections,
																	total_number_of_synapses,
																	neurons->d_membrane_potentials_v, 
																	d_synaptic_conductances_g);

	CudaCheckError();
}

void ConductanceSpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {

	conductance_update_synaptic_conductances_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(timestep, 
											d_synaptic_conductances_g, 
											d_synaptic_efficacies_or_weights, 
											d_time_of_last_spike_to_reach_synapse,
											d_biological_conductance_scaling_constants_lambda,
											total_number_of_synapses,
											current_time_in_seconds);

	CudaCheckError();

}

void ConductanceSpikingSynapses::update_presynaptic_activities(float timestep, float current_time_in_seconds) {

	conductance_update_presynaptic_activities_C_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_recent_presynaptic_activities_C,
							d_time_of_last_spike_to_reach_synapse,
							d_stdp,
							timestep,
							current_time_in_seconds,
							total_number_of_synapses);

	CudaCheckError();
}

void ConductanceSpikingSynapses::update_synaptic_efficacies_or_weights(float * d_recent_postsynaptic_activities_D, float timestep, float current_time_in_seconds, float * d_last_spike_time_of_each_neuron) {

	// printf("total_number_of_synapses: %d\n", total_number_of_synapses);

	conductance_update_synaptic_efficacies_or_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_recent_presynaptic_activities_C,
																d_recent_postsynaptic_activities_D,
																timestep,
																d_postsynaptic_neuron_indices,
																d_synaptic_efficacies_or_weights,
																current_time_in_seconds,
																d_time_of_last_spike_to_reach_synapse,
																d_last_spike_time_of_each_neuron,
																d_stdp,
																total_number_of_synapses);

	CudaCheckError();

}


__global__ void conductance_calculate_postsynaptic_current_injection_kernal(int * d_presynaptic_neuron_indices,
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


		// if (idx == 1000) printf("reversal_potential_Vhat: %f\n", reversal_potential_Vhat);

		float membrane_potential_v = d_membrane_potentials_v[postsynaptic_neuron_index];
		float synaptic_conductance_g = d_synaptic_conductances_g[idx];

		float component_for_sum = synaptic_conductance_g * (reversal_potential_Vhat - membrane_potential_v);
		if (component_for_sum != 0.0) {
			// if (idx == 1000) {
			// 	printf("membrane_potential_v: %1.12f\n", membrane_potential_v);
			// 	printf("synaptic_conductance_g: %1.12f\n", synaptic_conductance_g);
			// 	printf("component_for_sum, %1.12f\n", component_for_sum);
			// }
			atomicAdd(&d_neurons_current_injections[postsynaptic_neuron_index], component_for_sum);
		}

		idx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}



__global__ void conductance_update_synaptic_conductances_kernal(float timestep, 
														float * d_synaptic_conductances_g, 
														float * d_synaptic_efficacies_or_weights, 
														float * d_time_of_last_spike_to_reach_synapse,
														float * d_biological_conductance_scaling_constants_lambda,
														int total_number_of_synapses,
														float current_time_in_seconds) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapses) {

		float synaptic_conductance_g = d_synaptic_conductances_g[idx];
		float decay_term_tau_g = 0.004; // Is this the synaptic time constant?

		float new_conductance = (1.0 - (timestep/decay_term_tau_g)) * synaptic_conductance_g;
		// float new_conductance = synaptic_conductance_g;
		// if (idx == 10000) {
		// 	// printf("synaptic_conductance_g: %1.16f\n", synaptic_conductance_g);
		// 	printf("new_conductance: %1.16f\n", new_conductance);
		// }
		
		if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
			float timestep_times_synaptic_efficacy = timestep * d_synaptic_efficacies_or_weights[idx];
			float biological_conductance_scaling_constant_lambda = d_biological_conductance_scaling_constants_lambda[idx];
			float timestep_times_synaptic_efficacy_times_scaling_constant = timestep_times_synaptic_efficacy * biological_conductance_scaling_constant_lambda;
			new_conductance += timestep_times_synaptic_efficacy_times_scaling_constant;
			// if (idx == 10000) {
			// 	// printf("synaptic_conductance_g: %1.16f\n", synaptic_conductance_g);
			// 	printf("new_conductance: %1.16f\n", new_conductance);
			// }	

			// if (idx == 93000) {
			// 	printf("SPIKE ARRIVED! new_conductance: %1.16f\n", new_conductance);
			// 	// printf("SPIKE ARRIVED! timestep_times_synaptic_efficacy_times_scaling_constant: %1.16f\n", timestep_times_synaptic_efficacy_times_scaling_constant);
			// 	// printf("SPIKE ARRIVED! timestep_times_synaptic_efficacy: %1.12f\n", timestep_times_synaptic_efficacy);
			// 	// printf("SPIKE ARRIVED! timestep: %1.12f\n", timestep);
			// 	// printf("SPIKE ARRIVED! synaptic_efficacy: %1.12f\n", d_synaptic_efficacies_or_weights[idx]);
			// }
		}

		if (synaptic_conductance_g != new_conductance) {
			// printf("NEW CONDUCTANCE: %f\n", new_conductance);
			d_synaptic_conductances_g[idx] = new_conductance;
		}

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();

}


__global__ void conductance_update_presynaptic_activities_C_kernal(float* d_recent_presynaptic_activities_C,
							float* d_time_of_last_spike_to_reach_synapse,
							bool* d_stdp,
							float timestep, 
							float current_time_in_seconds,
							size_t total_number_of_synapses) {

	int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = t_idx;
	while (idx < total_number_of_synapses) {

		if (d_stdp[idx] == true) {

			float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];
			float decay_term_tau_C = 0.004; // Should be variable between 0.003 and 0.075

			float new_recent_presynaptic_activity_C = (1 - (timestep/decay_term_tau_C)) * recent_presynaptic_activity_C;

			if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
				float model_parameter_alpha_c = 0.5;
				new_recent_presynaptic_activity_C += timestep * model_parameter_alpha_c * (1 - recent_presynaptic_activity_C);
			}

			if (recent_presynaptic_activity_C != new_recent_presynaptic_activity_C) {
				d_recent_presynaptic_activities_C[idx] = new_recent_presynaptic_activity_C;
			}

		}

		idx += blockDim.x * gridDim.x;

	}

}


__global__ void conductance_update_synaptic_efficacies_or_weights_kernal(float * d_recent_presynaptic_activities_C,
																float * d_recent_postsynaptic_activities_D,
																float timestep,
																int* d_postsynaptic_neuron_indices,
																float* d_synaptic_efficacies_or_weights,
																float current_time_in_seconds,
																float * d_time_of_last_spike_to_reach_synapse,
																float * d_last_spike_time_of_each_neuron,
																bool* d_stdp,
																size_t total_number_of_synapses) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	while (idx < total_number_of_synapses) {

		if (d_stdp[idx] == true) {

			float synaptic_efficacy_delta_g = d_synaptic_efficacies_or_weights[idx];
			float new_synaptic_efficacy = synaptic_efficacy_delta_g;

			float new_componet = 0.0;

			int postsynaptic_neuron_index = d_postsynaptic_neuron_indices[idx];
			
			// if (idx == 1000) printf("d_last_spike_time_of_each_neuron[postsynaptic_neuron_index]: %f, current_time_in_seconds: %f\n", d_last_spike_time_of_each_neuron[postsynaptic_neuron_index], current_time_in_seconds);

			// if (idx == 1000) {
			// 	printf("d_recent_presynaptic_activities_C: %1.12f\n", d_recent_presynaptic_activities_C[idx]);
			// 	printf("d_recent_postsynaptic_activities_D: %1.12f\n\n", d_recent_postsynaptic_activities_D[postsynaptic_neuron_index]);
			// }

			if (d_last_spike_time_of_each_neuron[postsynaptic_neuron_index] == current_time_in_seconds) {
				float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];
				float new_componet_addition = ((1 - synaptic_efficacy_delta_g) * recent_presynaptic_activity_C);
				new_componet += new_componet_addition;
				// if (idx == 1000) printf("SPIKE AT POSTSYN: current_time_in_seconds: %f\n", current_time_in_seconds);
			}

			if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
				// if (idx == 1000) printf("SPIKE AT PRESYN: current_time_in_seconds: %f\n", current_time_in_seconds);
				float recent_postsynaptic_activity_D = d_recent_postsynaptic_activities_D[postsynaptic_neuron_index];
				new_componet -= (synaptic_efficacy_delta_g * recent_postsynaptic_activity_D);
			}			

			if (new_componet != 0.0) {
				// float decay_term_tau_delta_g = timestep / 0.1;
				// float learning_rate_rho = timestep/decay_term_tau_delta_g;
				// printf("learning_rate_rho: %f\n", learning_rate_rho);
				float learning_rate_rho = 0.1;
				new_componet = learning_rate_rho * new_componet;
				new_synaptic_efficacy += new_componet;
			}

			
			if (synaptic_efficacy_delta_g != new_synaptic_efficacy) {
				// if (idx == 1000) {
				// 	printf("synaptic_efficacy_delta_g: %1.12f\n", synaptic_efficacy_delta_g);
				// 	printf("new_synaptic_efficacy: %1.12f\n", new_synaptic_efficacy);
				// }

				new_synaptic_efficacy = max(new_synaptic_efficacy, 0.0);
				new_synaptic_efficacy = min(new_synaptic_efficacy, 1.0);

				d_synaptic_efficacies_or_weights[idx] = new_synaptic_efficacy;
			}

		}

		idx += blockDim.x * gridDim.x;
	}

}