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

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	TimerWithMessages * experiment_timer = new TimerWithMessages();

	bool command_line_arguments_passed = false;

	int optimisation_stage = 4;

	int number_of_optimisation_stages = 4;
	int number_of_new_parameters_per_optimisation_stage = 4;
	float optimisation_parameters[number_of_optimisation_stages][number_of_new_parameters_per_optimisation_stage];

	optimisation_parameters[0][0] = 2.5191751063e-03; //FF
	optimisation_parameters[0][1] = 8.8227295814e-05; //E2I
	optimisation_parameters[0][2] = 4.4264296185e-03; //I2E
	optimisation_parameters[0][3] = 2.6775065427e-03; //E2E
	optimisation_parameters[1][0] = 2.7736894247e-04;
	optimisation_parameters[1][1] = 8.9909369312e-05;
	optimisation_parameters[1][2] = 1.0000000000e-06;
	optimisation_parameters[1][3] = 3.9666270087e-03;
	optimisation_parameters[2][0] = 2.4246870746e-04;
	optimisation_parameters[2][1] = 8.3654704179e-05;
	optimisation_parameters[2][2] = 5.2362272268e-04;
	optimisation_parameters[2][3] = 4.3139460801e-03;
	optimisation_parameters[3][0] = 1.4086389598e-04;
	optimisation_parameters[3][1] = 6.4970154059e-05;
	optimisation_parameters[3][2] = 6.3505187529e-04;
	optimisation_parameters[3][3] = 1.2694665284e-03;

	// printf("argc = %d\n", argc);
	if (argc > 1) {
		command_line_arguments_passed = true;

		optimisation_stage = std::stoi(argv[3]);
		optimisation_parameters[optimisation_stage][0] = std::stof(argv[4]);
		optimisation_parameters[optimisation_stage][1] = std::stof(argv[5]);
		optimisation_parameters[optimisation_stage][2]= std::stof(argv[6]);
		optimisation_parameters[optimisation_stage][3] = std::stof(argv[7]);
		bool take_previous_layer_parameters_from_files = std::stoi(argv[8]);

		if (take_previous_layer_parameters_from_files) {

			std::string parameter_from_file_as_string;
			for (int i = 0; i < optimisation_stage; i++) {
				std::string file_name("../OptimalParameters/optimal_parameters_from_optimisation_iteration_" + std::to_string(i) + ".txt");
				// printf("file_name: %s\n", file_name.c_str());
				std::ifstream optimal_parameters_from_optimisation_iteration(file_name.c_str());
				for (int j = 0; j < number_of_new_parameters_per_optimisation_stage; j++) {
					std::getline(optimal_parameters_from_optimisation_iteration, parameter_from_file_as_string);
					// printf("parameter_from_file_as_string: %s\n", parameter_from_file_as_string.c_str());
					optimisation_parameters[i][j] = std::atof(parameter_from_file_as_string.c_str());
				}
			}
		}
	}

	
	// Create an instance of the Simulator and set the timestep
	Simulator simulator;
	float timestep = 0.0001;
	simulator.SetTimestep(timestep);
	simulator.high_fidelity_spike_storage = true;

	LIFSpikingNeurons * lif_spiking_neurons = new LIFSpikingNeurons();
	ImagePoissonInputSpikingNeurons* input_neurons = new ImagePoissonInputSpikingNeurons();
	ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
	EvansSTDP * evans_stdp = new EvansSTDP();

	/////////// STDP SETUP ///////////
	evans_stdp_parameters_struct * STDP_PARAMS = new evans_stdp_parameters_struct();
	evans_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (stdp_parameters_struct *) STDP_PARAMS);

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

	// input_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "../../MatlabGaborFilter/Inputs/", 100.0f);
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
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;			
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 50;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[0][0];
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.002;

	conductance_spiking_synapse_parameters_struct * E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;			
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 50;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[1][0];
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.002;

	conductance_spiking_synapse_parameters_struct * E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 30;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[0][1];
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.002;

	conductance_spiking_synapse_parameters_struct * I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 30;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[0][2];
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.005;

	conductance_spiking_synapse_parameters_struct * E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = 5.0*timestep;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = 3.0f*pow(10, -3);
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 20;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = optimisation_parameters[0][3];
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = 0.005;

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
	bool simulate_network_to_train_network = true;
	bool simulate_network_to_test_trained = true;
	float single_score_to_write_to_file_for_dakota_optimisation = 0.0;


	/////////// SIMULATE NETWORK TO TEST UNTRAINED ///////////
	float presentation_time_per_stimulus_per_epoch = 1.0f;
	bool record_spikes = false;
	bool save_recorded_spikes_to_file = false;
	int number_of_bins = 3;
	SpikeAnalyser * spike_analyser_for_untrained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonInputSpikingNeurons*)simulator.input_neurons);
	if (simulate_network_to_test_untrained) {

		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_to_file, spike_analyser_for_untrained_network);		
		
		spike_analyser_for_untrained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);
		spike_analyser_for_untrained_network->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		spike_analyser_for_untrained_network->calculate_combined_powered_distance_from_average_score();
		// single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score;
		single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2] + spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2 + 1];

		// simulator.recording_electrodes->delete_and_reset_recorded_spikes();

	}


	/////////// SIMULATE NETWORK TRAINING ///////////
	presentation_time_per_stimulus_per_epoch = 0.25f;
	int number_of_epochs = 10;
	bool present_stimuli_in_random_order = true;
	if (simulate_network_to_train_network) {
		simulator.RunSimulationToTrainNetwork(presentation_time_per_stimulus_per_epoch, number_of_epochs, present_stimuli_in_random_order);
	}



	/////////// SIMULATE NETWORK TO TEST TRAINED ///////////
	presentation_time_per_stimulus_per_epoch = 1.0f;
	record_spikes = false;
	save_recorded_spikes_to_file = false;
	if (simulate_network_to_test_trained) {
		SpikeAnalyser * spike_analyser_for_trained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonInputSpikingNeurons*)simulator.input_neurons);
		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_to_file, spike_analyser_for_trained_network);
		spike_analyser_for_trained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);

		single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_trained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores - spike_analyser_for_untrained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores;
		printf("single_score_to_write_to_file_for_dakota_optimisation: %f\n", single_score_to_write_to_file_for_dakota_optimisation);
		// string file = RESULTS_DIRECTORY + prefix_string + "_Epoch" + to_string(epoch_number) + "_" + to_string(clock());
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
