#include "CurrentSpikingSynapses.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"

void CurrentSpikingSynapses::AddGroup(int presynaptic_group_id, 
                                      int postsynaptic_group_id, 
                                      Neurons * neurons,
                                      Neurons * input_neurons,
                                      float timestep,
                                      synapse_parameters_struct * synapse_params) {
	
  SpikingSynapses::AddGroup(presynaptic_group_id, 
                            postsynaptic_group_id, 
                            neurons,
                            input_neurons,
                            timestep,
                            synapse_params);

}


void CurrentSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {

  // CUDA printf("number_of_synapse_blocks_per_grid.x: %d\n", number_of_synapse_blocks_per_grid.x);

        /*CUDA
	current_calculate_postsynaptic_current_injection_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_synaptic_efficacies_or_weights,
																	d_time_of_last_spike_to_reach_synapse,
																	d_postsynaptic_neuron_indices,
																	neurons->d_current_injections,
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
        */
}
