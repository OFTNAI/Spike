/*

	An Example Model for running the SPIKE simulator

	To create the executable for this network:
	- Run cmake from the build directory: "cmake ../"
	- Make this example: "make ExampleExperiment"
	- Finally, execute the binary: "./ExampleExperiment"


*/


// Import the SpikingModel and Simulator files
#include "Spike/Models/SpikingModel.hpp"
#include "Spike/Simulator/Simulator.hpp"

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	/*
			CHOOSE THE COMPONENTS OF YOUR SIMULATION
	*/

	// Create an instance of the Model
	SpikingModel* ExampleModel = new SpikingModel();
		

	// Set up the simulator with a timestep at which the neuron, synapse and STDP properties will be calculated 
	float timestep = 0.00002;  // In seconds
	ExampleModel->SetTimestep(timestep);


	// Choose an input neuron type
	GeneratorInputSpikingNeurons* generator_input_neurons = new GeneratorInputSpikingNeurons();
	// PoissonInputSpikingNeurons* input neurons = new PoissonInputSpikingNeurons();

	// Choose your neuron type
	LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
	// IzhikevichSpikingNeurons* izh_spiking_neurons = new IzhikevichSpikingNeurons();

	// Choose your synapse type
	CurrentSpikingSynapses * current_spiking_synapses = new CurrentSpikingSynapses();
	// ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();

	// Choose an STDP type, it must be initialized with a set of parameters
	evans_stdp_plasticity_parameters_struct * STDP_PARAMS = new evans_stdp_plasticity_parameters_struct();	// You can use the default Values
	EvansSTDPPlasticity * evans_stdp = new EvansSTDPPlasticity((SpikingSynapses *) current_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) generator_input_neurons, (stdp_plasticity_parameters_struct *) STDP_PARAMS);

	weightnorm_spiking_plasticity_parameters_struct * NORM_PARAMS = new weightnorm_spiking_plasticity_parameters_struct();
	NORM_PARAMS->settarget = true;
	NORM_PARAMS->target = 1.0f;
	WeightNormSpikingPlasticity * norm_plasticity = new WeightNormSpikingPlasticity((SpikingSynapses *) current_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) generator_input_neurons, (stdp_plasticity_parameters_struct *) NORM_PARAMS);

	// Allocate your chosen components to the simulator
	ExampleModel->input_spiking_neurons = generator_input_neurons;
	ExampleModel->spiking_neurons = lif_spiking_neurons;
	ExampleModel->spiking_synapses = current_spiking_synapses;
	ExampleModel->AddPlasticityRule(evans_stdp);
	ExampleModel->AddPlasticityRule(norm_plasticity);

	/*
			SETUP PROPERTIES AND CREATE NETWORK:
		
		Note: 
		All Neuron, Synapse and STDP types have associated parameters structures.
		These structures are defined in the header file for that class and allow us to set properties.
	*/

	// SETTING UP INPUT NEURONS
	// Creating an input neuron parameter structure
	generator_input_spiking_neuron_parameters_struct* input_neuron_params = new generator_input_spiking_neuron_parameters_struct();
	// Setting the dimensions of the input neuron layer
	input_neuron_params->group_shape[0] = 1;		// x-dimension of the input neuron layer
	input_neuron_params->group_shape[1] = 10;		// y-dimension of the input neuron layer
	// Create a group of input neurons. This function returns the ID of the input neuron group
	int input_layer_ID = ExampleModel->AddInputNeuronGroup(input_neuron_params);

	// We can now assign a set of spike times to neurons in the input layer
	int num_spikes = 5;
	int neuron_ids[5] = {0, 1, 3, 6, 7};
	float spike_times[5] = {0.1f, 0.3f, 0.2f, 0.5f, 0.9f};
	// Adding this stimulus to the input neurons
	generator_input_neurons->AddStimulus(num_spikes, neuron_ids, spike_times);


	// SETTING UP NEURON GROUPS
	// Creating an LIF parameter structure for an excitatory neuron population and an inhibitory
	// 1 x 100 Layer
	lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
	excitatory_population_params->group_shape[0] = 1;
	excitatory_population_params->group_shape[1] = 100;
	excitatory_population_params->resting_potential_v0 = -0.074f;
	excitatory_population_params->threshold_for_action_potential_spike = -0.053f;
	excitatory_population_params->somatic_capacitance_Cm = 500.0*pow(10, -12);
	excitatory_population_params->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);

	lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
	inhibitory_population_params->group_shape[0] = 1;
	inhibitory_population_params->group_shape[1] = 100;
	inhibitory_population_params->resting_potential_v0 = -0.082f;
	inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
	inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
	inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);

	// Create populations of excitatory and inhibitory neurons
	int excitatory_neuron_layer_ID = ExampleModel->AddNeuronGroup(excitatory_population_params);
	int inhibitory_neuron_layer_ID = ExampleModel->AddNeuronGroup(inhibitory_population_params);


	// SETTING UP SYNAPSES
	// Creating a synapses parameter structure for connections from the input neurons to the excitatory neurons
	spiking_synapse_parameters_struct* input_to_excitatory_parameters = new spiking_synapse_parameters_struct();
	input_to_excitatory_parameters->weight_range_bottom = 0.5f;		// Create uniform distributions of weights [0.5, 10.0]
	input_to_excitatory_parameters->weight_range_top = 10.0f;
	input_to_excitatory_parameters->delay_range[0] = timestep;		// Create uniform distributions of delays [1 timestep, 5 timesteps]
	input_to_excitatory_parameters->delay_range[1] = 5*timestep;
	// The connectivity types for synapses include:
		// CONNECTIVITY_TYPE_ALL_TO_ALL
		// CONNECTIVITY_TYPE_ONE_TO_ONE
		// CONNECTIVITY_TYPE_RANDOM
		// CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE
		// CONNECTIVITY_TYPE_SINGLE
	input_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	input_to_excitatory_parameters->plasticity_vec.push_back(evans_stdp);
	input_to_excitatory_parameters->plasticity_vec.push_back(norm_plasticity);

	// Creating a set of synapse parameters for connections from the excitatory neurons to the inhibitory neurons
	spiking_synapse_parameters_struct * excitatory_to_inhibitory_parameters = new spiking_synapse_parameters_struct();
	excitatory_to_inhibitory_parameters->weight_range_bottom = 10.0f;
	excitatory_to_inhibitory_parameters->weight_range_top = 10.0f;
	excitatory_to_inhibitory_parameters->delay_range[0] = 5.0*timestep;
	excitatory_to_inhibitory_parameters->delay_range[1] = 3.0f*pow(10, -3);
	excitatory_to_inhibitory_parameters->connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;
	excitatory_to_inhibitory_parameters->plasticity_vec.push_back(nullptr);

	// Creating a set of synapse parameters from the inhibitory neurons to the excitatory neurons
	spiking_synapse_parameters_struct * inhibitory_to_excitatory_parameters = new spiking_synapse_parameters_struct();
	excitatory_to_inhibitory_parameters->weight_range_bottom = -5.0f;
	excitatory_to_inhibitory_parameters->weight_range_top = -2.5f;
	inhibitory_to_excitatory_parameters->delay_range[0] = 5.0*timestep;
	inhibitory_to_excitatory_parameters->delay_range[1] = 3.0f*pow(10, -3);
	inhibitory_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	inhibitory_to_excitatory_parameters->plasticity_vec.push_back(nullptr);
	

	// CREATING SYNAPSES
	// When creating synapses, the ids of the presynaptic and postsynaptic populations are all that are required
	// Note: Input neuron populations cannot be post-synaptic on any synapse
	ExampleModel->AddSynapseGroup(input_layer_ID, excitatory_neuron_layer_ID, input_to_excitatory_parameters);
	ExampleModel->AddSynapseGroup(excitatory_neuron_layer_ID, inhibitory_neuron_layer_ID, excitatory_to_inhibitory_parameters);
	ExampleModel->AddSynapseGroup(inhibitory_neuron_layer_ID, excitatory_neuron_layer_ID, inhibitory_to_excitatory_parameters);

	// SETTING UP STDP
	// Getting the STDP parameter structure for this STDP type
	STDP_PARAMS->decay_term_tau_C = 0.015;
	STDP_PARAMS->decay_term_tau_D = 0.025;


	
	/*
			SETUP THE SIMULATOR
	*/
	// To complete the model set up:
	ExampleModel->finalise_model();	

	// The simulator shall now be initialised. It has a number of options that can be set up
	Simulator_Options* simoptions = new Simulator_Options();

	// Some examples of options that can be set up:
	// Set up amount of time per stimulus
	simoptions->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = 1.0f;
	// General options, e.g. turn on or off plasticity
	simoptions->run_simulation_general_options->apply_plasticity_to_relevant_synapses = true;
	// Presentation options (only applicable for multiple stimuli)
	simoptions->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS;
	// Recoding electrode options: Allow the saving of spike times in the simulation
	simoptions->recording_electrodes_options->collect_neuron_spikes_recording_electrodes_bool = true;
	simoptions->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool = true;
	simoptions->recording_electrodes_options->network_state_archive_recording_electrodes_bool = true;
	simoptions->recording_electrodes_options->network_state_archive_optional_parameters->human_readable_storage = true;
	// File storage options
	simoptions->file_storage_options->save_recorded_neuron_spikes_to_file = true;
	simoptions->file_storage_options->write_initial_synaptic_weights_to_file_bool = true;
	simoptions->file_storage_options->human_readable_storage = true;	
	
	// Finally, create and execute the simulation that you desire
	Simulator * simulator = new Simulator(ExampleModel, simoptions);
	simulator->RunSimulation();
	return 0;
}

