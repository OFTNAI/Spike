#include "SpikingModel.hpp"

#include "Spike/Helpers/TerminalHelpers.hpp"
#include "Spike/Backend/Context.hpp"


// SpikingModel Constructor
SpikingModel::SpikingModel () {

	timestep = 0.0001f;

	spiking_synapses = nullptr;
	spiking_neurons = nullptr;
	input_spiking_neurons = nullptr;

}


// SpikingModel Destructor
SpikingModel::~SpikingModel () {

}


void SpikingModel::SetTimestep(float timestep_parameter){

	if ((spiking_synapses == nullptr) || (spiking_synapses->total_number_of_synapses == 0)) {
		timestep = timestep_parameter;
	} else {
		print_message_and_exit("You must set the timestep before creating any synapses.");
	}
}


int SpikingModel::AddNeuronGroup(neuron_parameters_struct * group_params) {

	if (spiking_neurons == nullptr) print_message_and_exit("Please set neurons pointer before adding neuron groups.");

	int neuron_group_id = spiking_neurons->AddGroup(group_params);
	return neuron_group_id;

}


int SpikingModel::AddInputNeuronGroup(neuron_parameters_struct * group_params) {

	if (input_spiking_neurons == nullptr) print_message_and_exit("Please set input_neurons pointer before adding inputs groups.");

	int input_group_id = input_spiking_neurons->AddGroup(group_params);
	return input_group_id;

}


void SpikingModel::AddSynapseGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params) {

	if (spiking_synapses == nullptr) print_message_and_exit("Please set synapse pointer before adding synapses.");

	spiking_synapses->AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							spiking_neurons,
							input_spiking_neurons,
							timestep,
							synapse_params);
}


void SpikingModel::AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params) {

	for (int i = 0; i < input_spiking_neurons->total_number_of_groups; i++) {

		AddSynapseGroup(CORRECTED_PRESYNAPTIC_ID(i, true), 
							postsynaptic_group_id,
							synapse_params);

	}

}


void SpikingModel::AddPlasticityRule(Plasticity * plasticity_rule){
	// Adds the new STDP rule to the vector of STDP Rule
	plasticity_rule_vec.push_back(plasticity_rule);
	// Returns an ID corresponding to this STDP Rule
	// return(plasticity_rule_vec.size());
}


void SpikingModel::finalise_model() {
  init_backend();
}
  

void SpikingModel::init_backend() {

  Backend::init_global_context();
  context = Backend::get_current_context();

  #ifndef SILENCE_MODEL_SETUP
  TimerWithMessages* timer = new TimerWithMessages("Setting Up Network...\n");
  #endif

  context->params.threads_per_block_neurons = 512;
  context->params.threads_per_block_synapses = 512;

  // Provides order of magnitude speedup for LIF (All to all atleast). 
  // Because all synapses contribute to current_injection on every iteration, having all threads in a block accessing only 1 or 2 positions in memory causes massive slowdown.
  // Randomising order of synapses means that each block is accessing a larger number of points in memory.
  // if (temp_model_type == 1) spiking_synapses->shuffle_synapses();

  // NB All these also call prepare_backed for the initial state:
  spiking_synapses->init_backend(context);
  spiking_neurons->init_backend(context);
  input_spiking_neurons->init_backend(context);
  for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++){
	plasticity_rule_vec[plasticity_id]->init_backend(context);
  }

  #ifndef SILENCE_MODEL_SETUP
  timer->stop_timer_and_log_time_and_message("Network set up.", true);
  #endif
}


void SpikingModel::prepare_backend() {

  spiking_synapses->prepare_backend();
  context->params.maximum_axonal_delay_in_timesteps = spiking_synapses->maximum_axonal_delay_in_timesteps;
  spiking_neurons->prepare_backend();
  input_spiking_neurons->prepare_backend();
  for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++){
	plasticity_rule_vec[plasticity_id]->prepare_backend();
  }
}


void SpikingModel::reset_state() {
  spiking_synapses->reset_state();
  spiking_neurons->reset_state();
  input_spiking_neurons->reset_state();
  for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++){
	plasticity_rule_vec[plasticity_id]->reset_state();
  }
}


void SpikingModel::perform_per_timestep_model_instructions(float current_time_in_seconds, bool apply_plasticity_to_relevant_synapses){

	spiking_neurons->state_update(current_time_in_seconds, timestep);
	input_spiking_neurons->state_update(current_time_in_seconds, timestep);

	spiking_synapses->state_update(spiking_neurons, input_spiking_neurons, current_time_in_seconds, timestep);

	if (apply_plasticity_to_relevant_synapses){
		for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++)
			plasticity_rule_vec[plasticity_id]->state_update(current_time_in_seconds, timestep); // spiking_neurons, 
	}
}


