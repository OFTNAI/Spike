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
#include "../Neurons/ImagePoissonSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../SpikeAnalyser/GraphPlotter.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"
#include <string>
#include <fstream>

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	TimerWithMessages * experiment_timer = new TimerWithMessages();

	bool command_line_arguments_passed = false;

	int optimisation_stage = 4;

	float G2E_FF_biological_conductance_scaling_constant_lambda = 0.001732477223;
	float L1_E2I_L_biological_conductance_scaling_constant_lambda = 0.0002244875205;
	float L1_I2E_L_biological_conductance_scaling_constant_lambda = 0.002573465137;
	float L1_E2E_L_biological_conductance_scaling_constant_lambda = 0.002194632248;

	float L1_2_L2_E2E_FF_biological_conductance_scaling_constant_lambda = 0.000320288275;
	float L2_E2I_L_biological_conductance_scaling_constant_lambda = 0.0001493473745;
	float L2_I2E_L_biological_conductance_scaling_constant_lambda = 0.00332457261;
	float L2_E2E_L_biological_conductance_scaling_constant_lambda = 0.001317482035;
 
	float L2_2_L3_E2E_FF_biological_conductance_scaling_constant_lambda = 0.0001702463826;
	float L3_E2I_L_biological_conductance_scaling_constant_lambda = 0.0001034771659;
	float L3_I2E_L_biological_conductance_scaling_constant_lambda = 0.004517876013;
	float L3_E2E_L_biological_conductance_scaling_constant_lambda = 0.004139377891;

	//VALUES COPIED FROM 2
	float L3_2_L4_E2E_FF_biological_conductance_scaling_constant_lambda = 0.0002910494922;
	float L4_E2I_L_biological_conductance_scaling_constant_lambda = 0.0001888349535;
	float L4_I2E_L_biological_conductance_scaling_constant_lambda = 0.003112143311;
	float L4_E2E_L_biological_conductance_scaling_constant_lambda = 0.0008515281938;

	// printf("argc = %d\n", argc);
	if (argc > 1) {
		command_line_arguments_passed = true;

		optimisation_stage = std::stoi(argv[3]);
		printf("optimisation_stage: %d\n", optimisation_stage);

		float optimisation_variable_1 = std::stof(argv[4]);
		float optimisation_variable_2 = std::stof(argv[5]);
		float optimisation_variable_3 = std::stof(argv[6]);
		float optimisation_variable_4 = std::stof(argv[7]);
		printf("optimisation_variable_1: %f\n", optimisation_variable_1);
		printf("optimisation_variable_2: %f\n", optimisation_variable_2);
		printf("optimisation_variable_3: %f\n", optimisation_variable_3);
		printf("optimisation_variable_4: %f\n", optimisation_variable_4);

		if (optimisation_stage == 0) {
			G2E_FF_biological_conductance_scaling_constant_lambda = optimisation_variable_1;
			L1_E2I_L_biological_conductance_scaling_constant_lambda = optimisation_variable_2;
			L1_I2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_3;
			L1_E2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_4;
		}

		if (optimisation_stage == 1) {
			L1_2_L2_E2E_FF_biological_conductance_scaling_constant_lambda = optimisation_variable_1;
			L2_E2I_L_biological_conductance_scaling_constant_lambda = optimisation_variable_2;
			L2_I2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_3;
			L2_E2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_4;
		}

		if (optimisation_stage == 2) {
			L2_2_L3_E2E_FF_biological_conductance_scaling_constant_lambda = optimisation_variable_1;
			L3_E2I_L_biological_conductance_scaling_constant_lambda = optimisation_variable_2;
			L3_I2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_3;
			L3_E2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_4;
		}

		if (optimisation_stage == 3) {
			L3_2_L4_E2E_FF_biological_conductance_scaling_constant_lambda = optimisation_variable_1;
			L4_E2I_L_biological_conductance_scaling_constant_lambda = optimisation_variable_2;
			L4_I2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_3;
			L4_E2E_L_biological_conductance_scaling_constant_lambda = optimisation_variable_4;
		}

	}

	
	// Create an instance of the Simulator and set the timestep
	Simulator simulator;
	float timestep = 0.0001;
	simulator.SetTimestep(timestep);

	LIFSpikingNeurons * lif_spiking_neurons = new LIFSpikingNeurons();
	ImagePoissonSpikingNeurons* input_neurons = new ImagePoissonSpikingNeurons();
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

	input_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "../../MatlabGaborFilter/Inputs/", 100.0f);
	// input_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs/", 100.0f);
	image_poisson_spiking_neuron_parameters_struct * image_poisson_spiking_group_params = new image_poisson_spiking_neuron_parameters_struct();
	image_poisson_spiking_group_params->rate = 30.0f;
	input_neurons->AddGroupForEachGaborType(image_poisson_spiking_group_params);

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
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = G2E_FF_biological_conductance_scaling_constant_lambda;
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
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L1_2_L2_E2E_FF_biological_conductance_scaling_constant_lambda;
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
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L1_E2I_L_biological_conductance_scaling_constant_lambda;
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
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L1_I2E_L_biological_conductance_scaling_constant_lambda;
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
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L1_E2E_L_biological_conductance_scaling_constant_lambda;
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

		E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L1_2_L2_E2E_FF_biological_conductance_scaling_constant_lambda;
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L2_E2I_L_biological_conductance_scaling_constant_lambda;
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L2_I2E_L_biological_conductance_scaling_constant_lambda;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L2_E2E_L_biological_conductance_scaling_constant_lambda;

		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_2, E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, INHIBITORY_NEURONS_LAYER_2, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

	}

	if (optimisation_stage >= 2) {

		E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L2_2_L3_E2E_FF_biological_conductance_scaling_constant_lambda;
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L3_E2I_L_biological_conductance_scaling_constant_lambda;
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L3_I2E_L_biological_conductance_scaling_constant_lambda;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L3_E2E_L_biological_conductance_scaling_constant_lambda;

		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_3, E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, INHIBITORY_NEURONS_LAYER_3, E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	}

	if (optimisation_stage >= 3) {

		E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L3_2_L4_E2E_FF_biological_conductance_scaling_constant_lambda;
		E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L4_E2I_L_biological_conductance_scaling_constant_lambda;
		I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L4_I2E_L_biological_conductance_scaling_constant_lambda;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = L4_E2E_L_biological_conductance_scaling_constant_lambda;

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
	float presentation_time_per_stimulus_per_epoch = 0.05f;
	bool record_spikes = false;
	bool save_recorded_spikes_to_file = false;
	int number_of_bins = 3;
	if (simulate_network_to_test_untrained) {

		SpikeAnalyser * spike_analyser_for_untrained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonSpikingNeurons*)simulator.input_neurons);
		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_to_file, spike_analyser_for_untrained_network);		
		
		// spike_analyser_for_untrained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);
		spike_analyser_for_untrained_network->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		spike_analyser_for_untrained_network->calculate_combined_powered_distance_from_average_score();
		// single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score;
		single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2] + spike_analyser_for_trained_network->combined_powered_distance_from_average_score_for_each_neuron_group[optimisation_stage*2 + 1];


		// GraphPlotter *graph_plotter = new GraphPlotter();
		// graph_plotter->plot_untrained_vs_trained_single_cell_information_for_all_objects(spike_analyser_for_untrained_network, spike_analyser_for_trained_network);
		// graph_plotter->plot_all_spikes(simulator.recording_electrodes);

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
		SpikeAnalyser * spike_analyser_for_trained_network = new SpikeAnalyser(simulator.neurons, (ImagePoissonSpikingNeurons*)simulator.input_neurons);
		simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_to_file, spike_analyser_for_trained_network);
		// spike_analyser_for_trained_network->calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS_LAYER_4, number_of_bins);

		single_score_to_write_to_file_for_dakota_optimisation = spike_analyser_for_trained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores;

		// GraphPlotter *graph_plotter = new GraphPlotter();
		// graph_plotter->plot_untrained_vs_trained_single_cell_information_for_all_objects(spike_analyser_for_untrained_network, spike_analyser_for_trained_network);
		// graph_plotter->plot_all_spikes(simulator.recording_electrodes);

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
