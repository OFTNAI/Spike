#include "ConductanceSpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

// ConductanceSpikingSynapses Constructor
ConductanceSpikingSynapses::ConductanceSpikingSynapses() {
	synaptic_conductances_g = NULL;
	d_synaptic_conductances_g = NULL;

	recent_presynaptic_activities_C = NULL;
	d_recent_presynaptic_activities_C = NULL;
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
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						connectivity_parameters_struct * connectivity_params,
						float parameter,
						float parameter_two) {
	
	
	SpikingSynapses::AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							connectivity_type, 
							weight_range,
							delay_range,
							stdp_on,
							connectivity_params,
							parameter,
							parameter_two);

	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses-1; i++){
		synaptic_conductances_g[i] = 0.0f;
		recent_presynaptic_activities_C[i] = 0.0f;
	}

	// printf("HYE4\n");

}

void ConductanceSpikingSynapses::increment_number_of_synapses(int increment) {

	SpikingSynapses::increment_number_of_synapses(increment);

	synaptic_conductances_g = (float*)realloc(synaptic_conductances_g, total_number_of_synapses * sizeof(float));
	recent_presynaptic_activities_C = (float*)realloc(recent_presynaptic_activities_C, total_number_of_synapses * sizeof(float));

}


void ConductanceSpikingSynapses::allocate_device_pointers() {

	SpikingSynapses::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_synaptic_conductances_g, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_recent_presynaptic_activities_C, sizeof(float)*total_number_of_synapses));

}

void ConductanceSpikingSynapses::reset_synapse_spikes() {

	SpikingSynapses::reset_synapse_spikes();

	// CudaSafeCall(cudaMemset(d_synaptic_conductances_g, 0.0f, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMemcpy(d_synaptic_conductances_g, synaptic_conductances_g, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_recent_presynaptic_activities_C, recent_presynaptic_activities_C, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));

}


void ConductanceSpikingSynapses::shuffle_synapses() {
	SpikingSynapses::shuffle_synapses();
}


void ConductanceSpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	SpikingSynapses::set_threads_per_block_and_blocks_per_grid(threads);
	
}


__global__ void conductance_calculate_postsynaptic_current_injection_kernal(int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							size_t total_number_of_synapses,
							float * d_membrane_potentials_v,
							float * d_synaptic_conductances_g);


__global__ void conductance_update_synaptic_conductances_kernal(float timestep, 
													float * d_synaptic_conductances_g, 
													float * d_synaptic_efficacies_or_weights, 
													float * d_time_of_last_spike_to_reach_synapse,
													int total_number_of_synapses,
													float current_time_in_seconds);


__global__ void conductance_update_presynaptic_activities_C_kernal(float* d_recent_presynaptic_activities_C,
							float* d_time_of_last_spike_to_reach_synapse,
							bool* d_stdp,
							float timestep,
							float current_time_in_seconds,
							size_t total_number_of_synapses);

__global__ void conductance_update_synaptic_efficacies_or_weights_kernal(float * d_recent_presynaptic_activities_C,
																float * d_recent_postsynaptic_activities_D,
																float timestep,
																int* d_postsynaptic_neuron_indices,
																float* d_synaptic_efficacies_or_weights,
																float current_time_in_seconds,
																float * d_time_of_last_spike_to_reach_synapse,
																float * d_last_spike_time_of_each_neuron,
																bool* d_stdp,
																size_t total_number_of_synapses);



void ConductanceSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds) {

	conductance_calculate_postsynaptic_current_injection_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_postsynaptic_neuron_indices,
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

}


__global__ void conductance_calculate_postsynaptic_current_injection_kernal(int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							size_t total_number_of_synapses,
							float * d_membrane_potentials_v,
							float * d_synaptic_conductances_g){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		float temp_reversal_potential_Vhat = 0.0;
		float membrane_potential_v = d_membrane_potentials_v[d_postsynaptic_neuron_indices[idx]];
		float synaptic_conductance_g = d_synaptic_conductances_g[idx];

		float component_for_sum = synaptic_conductance_g * (temp_reversal_potential_Vhat - membrane_potential_v);
		if (component_for_sum != 0.0) {
			atomicAdd(&d_neurons_current_injections[d_postsynaptic_neuron_indices[idx]], component_for_sum);
		}

	}
	__syncthreads();
}



__global__ void conductance_update_synaptic_conductances_kernal(float timestep, 
														float * d_synaptic_conductances_g, 
														float * d_synaptic_efficacies_or_weights, 
														float * d_time_of_last_spike_to_reach_synapse, 
														int total_number_of_synapses,
														float current_time_in_seconds) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		float synaptic_conductance_g = d_synaptic_conductances_g[idx];
		float decay_term_tau_g = 0.01; // Is this the synaptic time constant?

		float new_conductance = (1 - (timestep/decay_term_tau_g)) * synaptic_conductance_g;
		if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
			new_conductance += timestep * d_synaptic_efficacies_or_weights[idx];
		}

		if (synaptic_conductance_g != new_conductance) {
			d_synaptic_conductances_g[idx] = new_conductance;
		}

	}

}


__global__ void conductance_update_presynaptic_activities_C_kernal(float* d_recent_presynaptic_activities_C,
							float* d_time_of_last_spike_to_reach_synapse,
							bool* d_stdp,
							float timestep, 
							float current_time_in_seconds,
							size_t total_number_of_synapses) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {

		if (d_stdp[idx] == true) {

			float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];
			float decay_term_tau_C = 0.01; // Should be variable between 0.003 and 0.075

			float new_recent_presynaptic_activity_C = (1 - (timestep/decay_term_tau_C)) * recent_presynaptic_activity_C;
			if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
				float model_parameter_alpha_c = 0.5;
				new_recent_presynaptic_activity_C += timestep * model_parameter_alpha_c * (1 - recent_presynaptic_activity_C);
			}

			if (recent_presynaptic_activity_C != new_recent_presynaptic_activity_C) {
				d_recent_presynaptic_activities_C[idx] = new_recent_presynaptic_activity_C;
			}

		}

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
	if (idx < total_number_of_synapses) {

		if (d_stdp[idx] == true) {

			float synaptic_efficacy_delta_g = d_synaptic_efficacies_or_weights[idx];
			float new_synaptic_efficacy = synaptic_efficacy_delta_g;

			float new_componet = 0.0;

			if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
				float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];
				new_componet += ((1 - synaptic_efficacy_delta_g) * recent_presynaptic_activity_C);
			}

			if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
				int postsynaptic_neuron_index = d_postsynaptic_neuron_indices[idx];
				float recent_postsynaptic_activity_D = d_recent_postsynaptic_activities_D[postsynaptic_neuron_index];
				new_componet -= (synaptic_efficacy_delta_g * recent_postsynaptic_activity_D);
			}

			float decay_term_tau_delta_g = 0.01; // Can't find value in paper
			if (new_componet != 0.0) {
				new_componet = (timestep/decay_term_tau_delta_g) * new_componet;
				new_synaptic_efficacy += new_componet;
			}

			if (synaptic_efficacy_delta_g != new_synaptic_efficacy) {
				d_synaptic_efficacies_or_weights[idx] = new_synaptic_efficacy;
			}

		}

	}

}