// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE='ConductanceExperiment1' EXPERIMENT_DIRECTORY='Experiments'  model -j8


#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/ImagePoissonSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include <time.h>
#include "../SpikeAnalyser/SpikeAnalyser.h"

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	
	clock_t begin_entire_experiment = clock();
	
	// Create an instance of the Simulator and set the timestep
	Simulator simulator;
	float timestep = 0.0001;
	simulator.SetTimestep(timestep);
	simulator.SetNeuronType(new LIFSpikingNeurons());
	// simulator.SetInputNeuronType(new PoissonSpikingNeurons());
	simulator.SetInputNeuronType(new ImagePoissonSpikingNeurons());
	simulator.SetSynapseType(new ConductanceSpikingSynapses());

	simulator.synapses->print_synapse_group_details = false;
	simulator.synapses->learning_rate_rho = 0.1;
	ConductanceSpikingSynapses * conductance_spiking_synapses = (ConductanceSpikingSynapses*)simulator.synapses;
	conductance_spiking_synapses->decay_term_tau_g = 0.004; //0.004 is arbitrary non-zero value
	conductance_spiking_synapses->decay_term_tau_C = 0.004; //0.004 is arbitrary non-zero value
	conductance_spiking_synapses->synaptic_neurotransmitter_concentration_alpha_C = 0.5;

	LIFSpikingNeurons * lif_spiking_neurons = (LIFSpikingNeurons*)simulator.neurons;
	lif_spiking_neurons->decay_term_tau_D = 0.07; //0.07 arbitrary non-zero value
	lif_spiking_neurons->model_parameter_alpha_D = 0.5;

	/////////// ADD INPUT NEURONS ///////////
	printf("Adding Input Neurons...\n");
	clock_t adding_input_neurons_start = clock();

	ImagePoissonSpikingNeurons* input_neurons = (ImagePoissonSpikingNeurons*)simulator.input_neurons;
	input_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs/", 1000.0f);
	image_poisson_spiking_neuron_parameters_struct * image_poisson_spiking_group_params = new image_poisson_spiking_neuron_parameters_struct();
	image_poisson_spiking_group_params->rate = 30.0f;
	input_neurons->AddGroupForEachGaborType(image_poisson_spiking_group_params);

	clock_t adding_input_neurons_end = clock();
	float adding_input_neurons_total_time = float(adding_input_neurons_end - adding_input_neurons_start) / CLOCKS_PER_SEC;
	printf("Input Neurons Added. Time Taken: %f\n", adding_input_neurons_total_time);
	print_line_of_dashes_with_blank_lines_either_side();

	// poisson_spiking_neuron_parameters_struct * poisson_spiking_group_params = new poisson_spiking_neuron_parameters_struct();
	// poisson_spiking_group_params->rate = 30.0f;


	/////////// ADD NEURONS ///////////
	printf("Adding Neurons...\n");
	clock_t adding_neurons_start = clock();

	//
	lif_spiking_neuron_parameters_struct * EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.074f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capcitance_Cm = 500.0*pow(10, -12);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);

	lif_spiking_neuron_parameters_struct * INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.082f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capcitance_Cm = 214.0*pow(10, -12);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);

	//
	int EXCITATORY_LAYER_SHAPE[] = {32, 32};
	int INHIBITORY_LAYER_SHAPE[] = {16, 16};

	// int temp_poisson_input_layer = simulator.AddInputNeuronGroup(poisson_spiking_group_params, EXCITATORY_LAYER_SHAPE);

	int EXCITATORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS, EXCITATORY_LAYER_SHAPE);
	// int EXCITATORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS, EXCITATORY_LAYER_SHAPE);
	// int EXCITATORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS, EXCITATORY_LAYER_SHAPE);
	// int EXCITATORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS, EXCITATORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS, INHIBITORY_LAYER_SHAPE);
	// int INHIBITORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS, INHIBITORY_LAYER_SHAPE);
	// int INHIBITORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS, INHIBITORY_LAYER_SHAPE);
	// int INHIBITORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS, INHIBITORY_LAYER_SHAPE);

	clock_t adding_neurons_end = clock();
	float adding_neurons_total_time = float(adding_neurons_end - adding_neurons_start) / CLOCKS_PER_SEC;
	printf("Neurons Added. Time taken: %f\n", adding_neurons_total_time);
	print_line_of_dashes_with_blank_lines_either_side();


	/////////// ADD SYNAPSES ///////////
	printf("Adding Synapses...\n");
	clock_t adding_synapses_start = clock();


	conductance_spiking_synapse_parameters_struct * G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;			
	G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 50;
	G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = 7.9 * pow(10, -4);
	G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
	G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;

	conductance_spiking_synapse_parameters_struct * E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;			
	E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 50;
	E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = 5.0 * pow(10, -5);
	E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
	E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;

	conductance_spiking_synapse_parameters_struct * E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 30;
	E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = 3.4 * pow(10, -5);
	E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;

	conductance_spiking_synapse_parameters_struct * I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 30;
	I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = 5.0 * pow(10, -4);
	I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);

	conductance_spiking_synapse_parameters_struct * I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 5;
	I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = 20;
	I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = 0.9 * pow(10, -2);
	I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = 10.0;
	I2I_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);

	//
	float INPUT_TO_EXCITATORY_DELAY_RANGE[] = {timestep, timestep};
	float EXCITATORY_TO_EXCITATORY_DELAY_RANGE[] = {5.0*timestep, 3.0f*pow(10, -3)};
	float EXCITATORY_TO_INHIBITORY_DELAY_RANGE[] = {5.0*timestep, 3.0f*pow(10, -3)};
	float INHIBITORY_TO_EXCITATORY_DELAY_RANGE[] = {5.0*timestep, 3.0f*pow(10, -3)};

	//
	simulator.AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS_LAYER_1, INPUT_TO_EXCITATORY_DELAY_RANGE, G2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	
	// simulator.AddSynapseGroup(temp_poisson_input_layer, EXCITATORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, false, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);


	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_2, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_3, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_4, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, E2E_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);


	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, INHIBITORY_NEURONS_LAYER_1, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, INHIBITORY_NEURONS_LAYER_2, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, INHIBITORY_NEURONS_LAYER_3, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_4, INHIBITORY_NEURONS_LAYER_4, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, E2I_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

	simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_1, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	// simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_4, EXCITATORY_NEURONS_LAYER_4, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, I2E_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	
	clock_t adding_synapses_end = clock();
	float adding_synapses_total_time = float(adding_synapses_end - adding_synapses_start) / CLOCKS_PER_SEC;
	printf("Synapses Added. Time taken: %f\n", adding_synapses_total_time);
	print_line_of_dashes_with_blank_lines_either_side();

	//
	int temp_model_type = 1;
	simulator.setup_network(temp_model_type);

	// 
	// int number_of_timesteps_per_device_spike_copy_check = 50;
	// int device_spike_store_size_multiple_of_total_neurons = 10;
	// float proportion_of_device_spike_store_full_before_copy = 0.8;
	int number_of_timesteps_per_device_spike_copy_check = 50;
	int device_spike_store_size_multiple_of_total_neurons = 50;
	float proportion_of_device_spike_store_full_before_copy = 0.5;
	simulator.setup_recording_electrodes_for_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);
	// simulator.setup_recording_electrodes_for_input_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);


	// TRAINING
	float presentation_time_per_stimulus_per_epoch = 0.25f;
	int number_of_epochs = 10;
	bool save_spikes = false;
	bool apply_stdp_to_relevant_synapses = true;
	bool count_spikes_per_neuron_for_single_cell_analysis = false;
	bool present_stimuli_in_random_order = true;
	simulator.RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, temp_model_type, save_spikes, apply_stdp_to_relevant_synapses, count_spikes_per_neuron_for_single_cell_analysis, present_stimuli_in_random_order);

	// TESTING
	SpikeAnalyser * spike_analyser = new SpikeAnalyser();
	presentation_time_per_stimulus_per_epoch = 1.0f;
	save_spikes = true;
	simulator.RunSimulationToCountNeuronSpikesForSingleCellAnalysis(presentation_time_per_stimulus_per_epoch, temp_model_type, save_spikes, spike_analyser);

	clock_t end_entire_experiment = clock();
	float timed_entire_experiment = float(end_entire_experiment - begin_entire_experiment) / CLOCKS_PER_SEC;
	printf("Entire Experiment Time: %f\n", timed_entire_experiment);
	print_line_of_dashes_with_blank_lines_either_side();



	return 1;
}