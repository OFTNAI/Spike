// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE='ConductanceExperiment1' EXPERIMENT_DIRECTORY='Experiments'  model -j8


#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../STDP/STDP.h"
#include "../STDP/EvansSTDP.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"
#include <string>
#include <fstream>
#include "../Plotting/Plotter.h"

#include <iostream>
using namespace std;

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	TimerWithMessages * experiment_timer = new TimerWithMessages();


	// Parameters
	int number_of_layers = 4;
	int max_number_of_connections_per_pair = 1;
	float gaussian_synapses_standard_deviation_FF = 10.0;
	float gaussian_synapses_standard_deviation_LAT = 5.0;
	float biological_conductance_scaling_constant_lambda_G2E_FF = 0.001;
	float biological_conductance_scaling_constant_lambda_E2E_FF = 0.00001;
	float biological_conductance_scaling_constant_lambda_E2E_L = 0.00001;

	bool simulate_network_to_test_untrained = true;
	bool simulate_network_to_train_network = true;
	bool simulate_network_to_test_trained = true;
	bool human_readable_storage = true;

	// Parameters for testing
	float presentation_time_per_stimulus_per_epoch_test = 0.5f;
	bool record_spikes_test = true;
	bool save_recorded_spikes_and_states_to_file_test = true;

	// Parameters for training
	float presentation_time_per_stimulus_per_epoch_train = 0.1f;
	int number_of_epochs_train = 10;

	// Parameters for Information Analysis
	int number_of_bins = 3;

	// Parameters related to Dakota Optimization
	int number_of_new_parameters_per_optimisation_stage = 2;


	// init parameters
	bool is_optimisation = false;
	float biological_conductance_scaling_constant_lambda[number_of_layers][number_of_new_parameters_per_optimisation_stage];
	int optimisation_stage = number_of_layers;
	bool isTrained=false;

	//init parameters
	for (int l = 0; l<number_of_layers; l++){
		biological_conductance_scaling_constant_lambda[l][0] = 0.00001; //E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
		biological_conductance_scaling_constant_lambda[l][1]= 0.00001; //I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
	}


	if (argc > 1) {
		is_optimisation = true;

		optimisation_stage = stoi(argv[3]);
		biological_conductance_scaling_constant_lambda[optimisation_stage][0] = stof(argv[4]); //E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
		biological_conductance_scaling_constant_lambda[optimisation_stage][1]= stof(argv[5]); //I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
	}


	// Load preoptimised parameters from files
	string parameter_from_file_as_string;
	for (int i = 0; i < optimisation_stage; i++) {
		string file_name_prefix = "";
		if(is_optimisation)
			file_name_prefix = "../OptimalParameters/optimal_parameters_from_optimisation_iteration_";
		else
			file_name_prefix = "Experiments/OptimalParameters/optimal_parameters_from_optimisation_iteration_";
		string file_name(file_name_prefix + to_string(i) + ".txt");

		ifstream optimal_parameters_from_optimisation_iteration(file_name.c_str());
		for (int j = 0; j < number_of_new_parameters_per_optimisation_stage; j++) {
			getline(optimal_parameters_from_optimisation_iteration, parameter_from_file_as_string);
			biological_conductance_scaling_constant_lambda[i][j] = atof(parameter_from_file_as_string.c_str());
		}
	}
	

	// Create an instance of the Simulator and set the timestep
	Simulator simulator;
	float timestep = 0.00002;
	simulator.SetTimestep(timestep);
	simulator.high_fidelity_spike_storage = true;

	LIFSpikingNeurons * lif_spiking_neurons = new LIFSpikingNeurons();
	ImagePoissonInputSpikingNeurons* input_neurons = new ImagePoissonInputSpikingNeurons();
	ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
	EvansSTDP * evans_stdp = new EvansSTDP();

	/////////// STDP SETUP ///////////
	evans_stdp_parameters_struct * STDP_PARAMS = new evans_stdp_parameters_struct();
	STDP_PARAMS->decay_term_tau_C = 0.015;
	STDP_PARAMS->decay_term_tau_D = 0.025;
	evans_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) input_neurons, (stdp_parameters_struct *) STDP_PARAMS);

	simulator.SetNeuronType(lif_spiking_neurons);
	simulator.SetInputNeuronType(input_neurons);
	simulator.SetSynapseType(conductance_spiking_synapses);
	simulator.SetSTDPType(evans_stdp);

	conductance_spiking_synapses->print_synapse_group_details = false;

	////////// SET UP STATES FOR RANDOM STATE MANAGER SINGLETON ///////////
	int random_states_threads_per_block_x = 128;
	int random_states_number_of_blocks_x = 64;
	RandomStateManager::instance()->set_up_random_states(random_states_threads_per_block_x, random_states_number_of_blocks_x, 9);


	/////////// ADD INPUT NEURONS ///////////
	TimerWithMessages * adding_input_neurons_timer = new TimerWithMessages("Adding Input Neurons...\n");

	if (is_optimisation)
		input_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "../../MatlabGaborFilter/Inputs/", 100.0f);
	else
		input_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs/", 100.0f);

	image_poisson_input_spiking_neuron_parameters_struct * image_poisson_input_spiking_group_params = new image_poisson_input_spiking_neuron_parameters_struct();
	image_poisson_input_spiking_group_params->rate = 30.0f;
	input_neurons->AddGroupForEachGaborType(image_poisson_input_spiking_group_params);

	adding_input_neurons_timer->stop_timer_and_log_time_and_message("Input Neurons Added.", true);


	/////////// ADD NEURONS ///////////
	TimerWithMessages * adding_neurons_timer = new TimerWithMessages("Adding Neurons...\n");

	lif_spiking_neuron_parameters_struct * EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[0] = 32;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[1] = 32;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.074f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capcitance_Cm = 500.0*pow(10, -12);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);


	lif_spiking_neuron_parameters_struct * INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[0] = 16;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[1] = 16;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.082f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capcitance_Cm = 214.0*pow(10, -12);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);

	int EXCITATORY_NEURONS_LAYER_1 = 0;
	int EXCITATORY_NEURONS_LAYER_2 = 0;
	int EXCITATORY_NEURONS_LAYER_3 = 0;
	int EXCITATORY_NEURONS_LAYER_4 = 0;
	int INHIBITORY_NEURONS_LAYER_1 = 0;
	int INHIBITORY_NEURONS_LAYER_2 = 0;
	int INHIBITORY_NEURONS_LAYER_3 = 0;
	int INHIBITORY_NEURONS_LAYER_4 = 0;
	if (optimisation_stage >= 0) {
		EXCITATORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
		INHIBITORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
	}

	if (optimisation_stage >= 1) {
		EXCITATORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
		INHIBITORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
	}

	if (optimisation_stage >= 2) {
		EXCITATORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
		INHIBITORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
	}

	if (optimisation_stage >= 3) {
		EXCITATORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
		INHIBITORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS);
	}

	adding_neurons_timer->stop_timer_and_log_time_and_message("Neurons Added.", true);


	/////////// ADD SYNAPSES ///////////
	TimerWithMessages * adding_synapses_timer = new TimerWithMessages("Adding Synapses...\n");






	conductance_spiking_synapse_parameters_struct * G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = max_number_of_connections_per_pair;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 50;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.15;

	conductance_spiking_synapse_parameters_struct * E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = max_number_of_connections_per_pair;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 50;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_FF;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_FF;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.15;

	conductance_spiking_synapse_parameters_struct * E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = max_number_of_connections_per_pair;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 30;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[0][0];
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_LAT;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.002;

	conductance_spiking_synapse_parameters_struct * I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 30;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[0][1];
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_LAT;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.005;

	conductance_spiking_synapse_parameters_struct * E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 20;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_L;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_LAT;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.005;

	if (optimisation_stage >= 0) {
		simulator.AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS_LAYER_1, G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, INHIBITORY_NEURONS_LAYER_1, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_1, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_1, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	}


	if (optimisation_stage >= 1) {

//		cout<<"test: optimisation_parameters[1][0]:"<<optimisation_parameters[1][0]<<" optimisation_parameters[1][1]:" <<optimisation_parameters[1][1] << " optimisation_parameters[1][2]:"<<optimisation_parameters[1][2]<<" optimisation_parameters[1][3]:"<<optimisation_parameters[1][3]<<endl;

		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[1][0];
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[1][1];

		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_2, E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, INHIBITORY_NEURONS_LAYER_2, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

	}

	if (optimisation_stage >= 2) {
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[2][0];
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[2][1];

		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_3, E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, INHIBITORY_NEURONS_LAYER_3, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	}

	if (optimisation_stage >= 3) {
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[3][0];
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda[3][1];

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

	float single_score_to_write_to_file_for_dakota_optimisation_excit = 0.0;
	float single_score_to_write_to_file_for_dakota_optimisation_inhib = 0.0;


	/////////// SIMULATE NETWORK TO TEST UNTRAINED ///////////
	float presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_test;
	bool record_spikes = record_spikes_test;
	bool save_recorded_spikes_and_states_to_file = save_recorded_spikes_and_states_to_file_test;


	SpikeAnalyser * spike_analyser_for_untrained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonInputSpikingNeurons*)simulator.input_neurons);
	if (simulate_network_to_test_untrained) {
		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_and_states_to_file, spike_analyser_for_untrained_network,human_readable_storage,isTrained);
		
		spike_analyser_for_untrained_network->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		spike_analyser_for_untrained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);
		isTrained = true;
	}


	/////////// SIMULATE NETWORK TRAINING ///////////
	if (simulate_network_to_train_network) {
		presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_train;
		int stimulus_presentation_order_seed = 1;
		int number_of_epochs = number_of_epochs_train;
		Stimuli_Presentation_Struct * stimuli_presentation_params = new Stimuli_Presentation_Struct();
		stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
		stimuli_presentation_params->object_order = OBJECT_ORDER_ORIGINAL;//OBJECT_ORDER_RANDOM;
		stimuli_presentation_params->transform_order = TRANSFORM_ORDER_RANDOM;

		simulator.RunSimulationToTrainNetwork(presentation_time_per_stimulus_per_epoch, number_of_epochs, stimuli_presentation_params, stimulus_presentation_order_seed);
	}



	/////////// SIMULATE NETWORK TO TEST TRAINED ///////////
	SpikeAnalyser * spike_analyser_for_trained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonInputSpikingNeurons*)simulator.input_neurons);
	if (simulate_network_to_test_trained) {
		presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_test;
		record_spikes = record_spikes_test;
		save_recorded_spikes_and_states_to_file = save_recorded_spikes_and_states_to_file_test;

		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_and_states_to_file, spike_analyser_for_trained_network,human_readable_storage,isTrained);

		spike_analyser_for_trained_network->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		spike_analyser_for_trained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);
	}

	/////////// PLOT INFOANALYSIS RESULTS //////////////////
	if (simulate_network_to_test_untrained && simulate_network_to_test_trained){
		Plotter * plotter = new Plotter();
		plotter->plot_single_cell_information_analysis(spike_analyser_for_untrained_network, spike_analyser_for_trained_network);

	}


	/////////// WRITE NETWORK SCORE TO RESULTS FILE FOR DAKOTA OPTIMISATION ///////////
	if (is_optimisation){
		//output combined powered distance as a objective function of the optimization
		spike_analyser_for_untrained_network->calculate_combined_powered_distance_from_average_score();
		single_score_to_write_to_file_for_dakota_optimisation_excit = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2];
		single_score_to_write_to_file_for_dakota_optimisation_inhib = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2 + 1];

		TimerWithMessages * writing_network_score_to_results_file_timer = new TimerWithMessages("Writing Network Score to Results File for Dakota Optimisation...\n");

		float combined_information_score_training_increase = spike_analyser_for_trained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores - spike_analyser_for_untrained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores;
		printf("combined_information_score_training_increase: %f\n", combined_information_score_training_increase);

		ofstream resultsfile;
		resultsfile.open(argv[1], ios::out | ios::binary);
	//	resultsfile << to_string(single_score_to_write_to_file_for_dakota_optimisation_excit) << endl;
	//	resultsfile << to_string(single_score_to_write_to_file_for_dakota_optimisation_inhib) << endl;
		resultsfile << to_string(single_score_to_write_to_file_for_dakota_optimisation_excit + single_score_to_write_to_file_for_dakota_optimisation_inhib) << endl;
		resultsfile << to_string(combined_information_score_training_increase)<<endl;
		resultsfile.close();

		writing_network_score_to_results_file_timer->stop_timer_and_log_time_and_message("Network Score Written to File.", true);
	}


	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	return 0;
}
//
