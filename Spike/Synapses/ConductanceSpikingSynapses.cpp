#include "ConductanceSpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

// ConductanceSpikingSynapses Destructor
ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
  free(synaptic_conductances_g);
  free(biological_conductance_scaling_constants_lambda);
  free(reversal_potentials_Vhat);
  free(decay_terms_tau_g);
}


void ConductanceSpikingSynapses::AddGroup(int presynaptic_group_id, 
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

  conductance_spiking_synapse_parameters_struct * conductance_spiking_synapse_group_params = (conductance_spiking_synapse_parameters_struct*)synapse_params;

  for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses-1; i++){
    synaptic_conductances_g[i] = 0.0f;
    biological_conductance_scaling_constants_lambda[i] = conductance_spiking_synapse_group_params->biological_conductance_scaling_constant_lambda;
    reversal_potentials_Vhat[i] = conductance_spiking_synapse_group_params->reversal_potential_Vhat;
    decay_terms_tau_g[i] = conductance_spiking_synapse_group_params->decay_term_tau_g;
  }

}

void ConductanceSpikingSynapses::increment_number_of_synapses(int increment) {

  SpikingSynapses::increment_number_of_synapses(increment);

  synaptic_conductances_g = (float*)realloc(synaptic_conductances_g, total_number_of_synapses * sizeof(float));
  biological_conductance_scaling_constants_lambda = (float*)realloc(biological_conductance_scaling_constants_lambda, total_number_of_synapses * sizeof(float));
  reversal_potentials_Vhat = (float*)realloc(reversal_potentials_Vhat, total_number_of_synapses * sizeof(float));
  decay_terms_tau_g = (float*)realloc(decay_terms_tau_g, total_number_of_synapses * sizeof(float));

}


void ConductanceSpikingSynapses::reset_state() {
  SpikingSynapses::reset_state();
  backend()->reset_state();
}


void ConductanceSpikingSynapses::shuffle_synapses() {
  SpikingSynapses::shuffle_synapses();

  float * temp_synaptic_conductances_g = (float *)malloc(total_number_of_synapses*sizeof(float));
  float * temp_biological_conductance_scaling_constants_lambda = (float *)malloc(total_number_of_synapses*sizeof(float));
  float * temp_reversal_potentials_Vhat = (float *)malloc(total_number_of_synapses*sizeof(float));
  float * temp_decay_terms_tau_g = (float*)malloc(total_number_of_synapses*sizeof(float));

  for(int i = 0; i < total_number_of_synapses; i++) {

    temp_synaptic_conductances_g[i] = synaptic_conductances_g[original_synapse_indices[i]];
    temp_biological_conductance_scaling_constants_lambda[i] = biological_conductance_scaling_constants_lambda[original_synapse_indices[i]];
    temp_reversal_potentials_Vhat[i] = reversal_potentials_Vhat[original_synapse_indices[i]];
    temp_decay_terms_tau_g[i] = decay_terms_tau_g[original_synapse_indices[i]];
  }

  synaptic_conductances_g = temp_synaptic_conductances_g;
  biological_conductance_scaling_constants_lambda = temp_biological_conductance_scaling_constants_lambda;
  reversal_potentials_Vhat = temp_reversal_potentials_Vhat;
  decay_terms_tau_g = temp_decay_terms_tau_g;

}


void ConductanceSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {
  backend()->calculate_postsynaptic_current_injection(neurons, current_time_in_seconds, timestep);
  // After injecting current, update the conductances
  update_synaptic_conductances(timestep, current_time_in_seconds);
}

void ConductanceSpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {
  backend()->update_synaptic_conductances(timestep, current_time_in_seconds);
}

MAKE_PREPARE_BACKEND(ConductanceSpikingSynapses);
