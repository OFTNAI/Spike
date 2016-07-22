/*

	An Example Model for running the SPIKE simulator

	Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

	To create the executable for this network, run:
	make FILE='ExampleExperiment' EXPERIMENT_DIRECTORY='Experiments'  model -j8

	To create your own simulation, simply create a .cpp file similar to this one (with the network structure you desire) and run:
	make FILE='YOUREXPERIMENTFILENAME' EXPERIMENT_DIRECTORY='Experiments'  model -j8


*/



// The Simulator Class
#include "../Simulator/Simulator.h"							// The simulator class takes references to your neuron/synapse populations and runs the simulation

// Input Neuron Classes
#include "../Neurons/SpikingNeurons.h"						// The Spiking Neurons parent class is used when passing references to the simulator
#include "../Neurons/GeneratorInputSpikingNeurons.h"		// The Generator Input neuron type allows you to load and give an input neuron population specific spike times
// #include "../Neurons/PoissonInputSpikingNeurons.h"		// Poisson Input Neurons allow you to set the Poisson Rate of the population of input neurons

// Neuron Classes
#include "../Neurons/SpikingSynapses.h"						// The Spiking Synapses parent class is used when passing references to the simulator
#include "../Neurons/LIFSpikingNeurons.h"					// Leaky Integrate and Fire Implementation
// #include "../Neurons/IzhikevichSpikingNeurons.h"			// Izhikevich Spiking Neuron Implementation


// Synapse Classes
#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/CurrentSpikingSynapses.h"				// Current Spiking Synapses inject a square wave of current into a post-synaptic neuron when that synapse is active	
// #include "../Synapses/ConductanceSpikingSynapses.h"		// Conductance Spiking Synapses have a decaying conductance associated with synapses to inject current into post-synaptic neurons with some decay

// STDP Class
#include "../STDP/STDP.h"									// STDP class used to pass references to the simulator
#include "../STDP/Higgins.h"								// STDP rule used by Higgins in: http://biorxiv.org/content/early/2016/06/17/059428
// #include "../STDP/EvansSTDP.h"							// STDP rule used by Evans in: http://www.ncbi.nlm.nih.gov/pubmed/22848199

// Spike Analyser class for information analyses
#include "../SpikeAnalyser/SpikeAnalyser.h"

// Other helper code
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/TimerWithMessages.h"



// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	/*
			CHOOSE THE COMPONENTS OF YOUR SIMULATION
	*/

	// Create an instance of the Simulator
	Simulator simulator;

	// Set up the simulator with a timestep at which the neuron, synapse and STDP properties will be calculated 
	float timestep = 0.00002;  // In seconds
	simulator.SetTimestep(timestep);


	// Choose an input neuron type
	GeneratorInputSpikingNeurons* generator_input_neurons = new GeneratorInputSpikingNeurons();
	// PoissonInputSpikingNeurons* input neurons = new PoissonInputSpikingNeurons();

	// Choose your neuron type
	LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
	// IzhikevichSpikingNeurons* izh_spiking_neurons = new IzhikevichSpikingNeurons();

	// Choose your synapse type
	CurrentSpikingSynapses * current_spiking_synapses = new CurrentSpikingSynapses();
	// ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();

	// Choose an STDP type
	HigginsSTDP* higgins_stdp = new HigginsSTDP();
	// EvansSTDP * evans_stdp = new EvansSTDP();

	// Allocate your chosen components to the simulator
	simulator.SetInputNeuronType(generator_input_neurons);
	simulator.SetNeuronType(lif_spiking_neurons);
	simulator.SetSynapseType(current_spiking_synapses);
	simulator.SetSTDPType(higgins_stdp);


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
	int input_layer_ID = input_neurons.AddGroup(input_neuron_params);

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
	excitatory_population_params->somatic_capcitance_Cm = 500.0*pow(10, -12);
	excitatory_population_params->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);

	lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
	inhibitory_population_params->group_shape[0] = 1;
	inhibitory_population_params->group_shape[1] = 25;
	inhibitory_population_params->resting_potential_v0 = -0.082f;
	inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
	inhibitory_population_params->somatic_capcitance_Cm = 214.0*pow(10, -12);
	inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);

	// Create populations of excitatory and inhibitory neurons
	excitatory_neuron_layer_ID = simulator.AddNeuronGroup(excitatory_population_params);
	inhibitory_neuron_layer_ID = simulator.AddNeuronGroup(inhibitory_population_params);


	// SETTING UP SYNAPSES
	// Creating a synapses parameter structure for connections from the input neurons to the excitatory neurons
	current_spiking_synapse_parameters_struct* input_to_excitatory_parameters = new current_spiking_synapse_parameters_struct();
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
	input_to_excitatory_parameters->stdp_on = true;

	// Creating a set of synapse parameters for connections from the excitatory neurons to the inhibitory neurons
	current_spiking_synapse_parameters_struct * excitatory_to_inhibitory = new current_spiking_synapse_parameters_struct();
	excitatory_to_inhibitory_parameters->weight_range_bottom = 10.0f;
	excitatory_to_inhibitory_parameters->weight_range_top = 10.0f;
	excitatory_to_inhibitory_parameters->delay_range[0] = 5.0*timestep;
	excitatory_to_inhibitory_parameters->delay_range[1] = 3.0f*pow(10, -3);
	excitatory_to_inhibitory_parameters->connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;
	excitatory_to_inhibitory_parameters->stdp_on = false;

	// Creating a set of synapse parameters from the inhibitory neurons to the excitatory neurons
	conductance_spiking_synapse_parameters_struct * inhibitory_to_excitatory_parameters = new current_spiking_synapse_parameters_struct();
	excitatory_to_inhibitory_parameters->weight_range_bottom = -10.0f;
	excitatory_to_inhibitory_parameters->weight_range_top = -5.0f;
	inhibitory_to_excitatory_parameters->delay_range[0] = 5.0*timestep;
	inhibitory_to_excitatory_parameters->delay_range[1] = 3.0f*pow(10, -3);
	inhibitory_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	inhibitory_to_excitatory_parameters->stdp_on = false;
	

	// CREATING SYNAPSES
	// When creating synapses, input neurons must be treated differently and hence there is a unique function for them.
	// Input neurons can also only be 
	simulator.AddSynapseGroupFromInputNeurons()

	// SETTING UP STDP
	// Getting the STDP parameter structure for this STDP type
	higgins_stdp_parameters_struct * STDP_PARAMS = new higgins_stdp_parameters_struct();	// You can use the default Values
	higgins_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) izhikevich_spiking_neurons, (SpikingNeurons *) input_neurons, (stdp_parameters_struct *) STDP_PARAMS);
	// evans_stdp_parameters_struct * STDP_PARAMS = new evans_stdp_parameters_struct(); 	// Or Define the parameters of the STDP model
	// STDP_PARAMS->decay_term_tau_C = 0.015;
	// STDP_PARAMS->decay_term_tau_D = 0.025;
	// evans_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) input_neurons, (stdp_parameters_struct *) STDP_PARAMS);





	if (optimisation_stage >= 0) {
		simulator.AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS_LAYER_1, G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, INHIBITORY_NEURONS_LAYER_1, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_1, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_1, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	}


	if (optimisation_stage >= 1) {

		E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[1][0];
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[1][1];
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[1][2];
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[1][3];

		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_2, E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, INHIBITORY_NEURONS_LAYER_2, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

	}

	if (optimisation_stage >= 2) {

		E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[2][0];
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[2][1];
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[2][2];
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[2][3];

		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_3, E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, INHIBITORY_NEURONS_LAYER_3, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	}

	if (optimisation_stage >= 3) {

		E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[3][0];
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[3][1];
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[3][2];
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[3][3];

		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_4, E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_4, INHIBITORY_NEURONS_LAYER_4, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_4, EXCITATORY_NEURONS_LAYER_4, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_4, EXCITATORY_NEURONS_LAYER_4, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

	}
	
	

	
	adding_synapses_timer->stop_timer_and_log_time_and_message("Synapses Added.", true);


	/////////// SETUP NETWORK ///////////
	simulator.setup_network();




	/////////// SETUP RECORDING ELECTRODES ///////////
	int number_of_timesteps_per_device_spike_copy_check = 50;
	int device_spike_store_size_multiple_of_total_neurons = 52;
	float proportion_of_device_spike_store_full_before_copy = 0.2;
	simulator.setup_recording_electrodes_for_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);
	// simulator.setup_recording_electrodes_for_input_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);



	bool simulate_network_to_test_untrained = true;
	bool simulate_network_to_train_network = false;
	bool simulate_network_to_test_trained = false;
	float single_score_to_write_to_file_for_dakota_optimisation = 0.0;


	/////////// SIMULATE NETWORK TO TEST UNTRAINED ///////////
	float presentation_time_per_stimulus_per_epoch = 0.25f;
	bool record_spikes = false;
	bool save_recorded_spikes_to_file = false;
	int number_of_bins = 3;
	SpikeAnalyser * spike_analyser_for_untrained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonInputSpikingNeurons*)simulator.input_neurons);
	if (simulate_network_to_test_untrained) {

		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_to_file, spike_analyser_for_untrained_network);		
		
		spike_analyser_for_untrained_network->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		spike_analyser_for_untrained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);
		spike_analyser_for_untrained_network->calculate_combined_powered_distance_from_average_score();
		// single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score;
		single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2] + spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2 + 1];

	}


	/////////// SIMULATE NETWORK TRAINING ///////////
	presentation_time_per_stimulus_per_epoch = 0.01f;
	int stimulus_presentation_order_seed = 1;
	int number_of_epochs = 1;
	bool present_stimuli_in_random_order = true;
	Stimuli_Presentation_Struct * stimuli_presentation_params = new Stimuli_Presentation_Struct();
	stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	stimuli_presentation_params->object_order = OBJECT_ORDER_RANDOM;
	stimuli_presentation_params->transform_order = TRANSFORM_ORDER_RANDOM;
	if (simulate_network_to_train_network) {
		simulator.RunSimulationToTrainNetwork(presentation_time_per_stimulus_per_epoch, number_of_epochs, stimuli_presentation_params, stimulus_presentation_order_seed);
	}



	/////////// SIMULATE NETWORK TO TEST TRAINED ///////////
	presentation_time_per_stimulus_per_epoch = 0.25f;
	record_spikes = false;
	save_recorded_spikes_to_file = false;
	if (simulate_network_to_test_trained) {
		SpikeAnalyser * spike_analyser_for_trained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonInputSpikingNeurons*)simulator.input_neurons);
		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_to_file, spike_analyser_for_trained_network);

		spike_analyser_for_trained_network->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		spike_analyser_for_trained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);

		Plotter * plotter = new Plotter();
		plotter->plot_single_cell_information_analysis(spike_analyser_for_untrained_network, spike_analyser_for_trained_network);
	}


	/////////// WRITE NETWORK SCORE TO RESULTS FILE FOR DAKOTA OPTIMISATION ///////////
	TimerWithMessages * writing_network_score_to_results_file_timer = new TimerWithMessages("Writing Network Score to Results File for Dakota Optimisation...\n");
	// float combined_information_score_training_increase = spike_analyser_for_trained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores - spike_analyser_for_untrained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores;
	// printf("combined_information_score_training_increase: %f\n", combined_information_score_training_increase);
	std::ofstream resultsfile;
	resultsfile.open(argv[1], std::ios::out | std::ios::binary);
	resultsfile << std::to_string(single_score_to_write_to_file_for_dakota_optimisation) << std::endl;
	resultsfile.close();

	writing_network_score_to_results_file_timer->stop_timer_and_log_time_and_message("Network Score Written to File.", true);


	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	return 0;
}
//