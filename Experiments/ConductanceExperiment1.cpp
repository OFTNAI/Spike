// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE='ConductanceExperiment1' EXPERIMENT_DIRECTORY='Experiments'  model -j8


#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/ConductanceSpikingNeurons.h"
#include "../Neurons/ImagePoissonSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include <time.h>

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	clock_t begin_entire_experiment = clock();

	// Set the time_stepep of the simulation as required (time_stepep is measure in seconds)
	
	// Create an instance of the Simulator and set the time_step
	Simulator simulator;
	float time_step = 0.0001;
	simulator.SetTimestep(time_step);
	simulator.SetNeuronType(new ConductanceSpikingNeurons());
	// simulator.SetInputNeuronType(new PoissonSpikingNeurons());
	simulator.SetInputNeuronType(new ImagePoissonSpikingNeurons());
	simulator.SetSynapseType(new ConductanceSpikingSynapses());

	simulator.synapses->print_synapse_group_details = false;

	/////////// ADD INPUT NEURONS ///////////
	printf("Adding Input Neurons...\n");
	clock_t adding_input_neurons_start = clock();

	ImagePoissonSpikingNeurons* input_neurons = (ImagePoissonSpikingNeurons*)simulator.input_neurons;
	input_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs/");
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
	conductance_spiking_neuron_parameters_struct * EXCITATORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS = new conductance_spiking_neuron_parameters_struct();
	EXCITATORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->resting_potential_v0 = -0.074f;
	EXCITATORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	EXCITATORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->somatic_capcitance_Cm = 500.0*pow(10, -12);
	EXCITATORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);
	EXCITATORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->reversal_potential_Vhat = 0.0;

	conductance_spiking_neuron_parameters_struct * INHIBITORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS = new conductance_spiking_neuron_parameters_struct();
	INHIBITORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->resting_potential_v0 = -0.082f;
	INHIBITORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	INHIBITORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->somatic_capcitance_Cm = 214.0*pow(10, -12);
	INHIBITORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
	INHIBITORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS->reversal_potential_Vhat = -70.0*pow(10, -3);

	//
	int EXCITATORY_LAYER_SHAPE[] = {32, 32};
	int INHIBITORY_LAYER_SHAPE[] = {16, 16};

	// int temp_poisson_input_layer = simulator.AddInputNeuronGroup(poisson_spiking_group_params, EXCITATORY_LAYER_SHAPE);

	int EXCITATORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(EXCITATORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS, EXCITATORY_LAYER_SHAPE);
	// int EXCITATORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(conductance_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	// int EXCITATORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(conductance_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	// int EXCITATORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(conductance_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(INHIBITORY_CONDUCTANCE_SPIKING_NUERON_GROUP_PARAMS, INHIBITORY_LAYER_SHAPE);
	// int INHIBITORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(conductance_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	// int INHIBITORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(conductance_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	// int INHIBITORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(conductance_spiking_group_params, INHIBITORY_LAYER_SHAPE);

	clock_t adding_neurons_end = clock();
	float adding_neurons_total_time = float(adding_neurons_end - adding_neurons_start) / CLOCKS_PER_SEC;
	printf("Neurons Added. Time taken: %f\n", adding_neurons_total_time);
	print_line_of_dashes_with_blank_lines_either_side();


	/////////// ADD SYNAPSES ///////////
	printf("Adding Synapses...\n");
	clock_t adding_synapses_start = clock();

	float CONNECTIVITY_STANDARD_DEVIATION_SIGMA = 10.0;

	synapse_parameters_struct * synapse_parameters = new synapse_parameters_struct();
	synapse_parameters->max_number_of_connections_per_pair = 5;

	//
	float INPUT_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 1.0};
	float EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 1.0};
	float EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE[] = {0.0, 1.0};
	float INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 1.0};

	//
	float INPUT_TO_EXCITATORY_DELAY_RANGE[] = {time_step, time_step};
	float EXCITATORY_TO_EXCITATORY_DELAY_RANGE[] = {time_step, 5.0f*pow(10, -3)};
	float EXCITATORY_TO_INHIBITORY_DELAY_RANGE[] = {time_step, 5.0f*pow(10, -3)};
	float INHIBITORY_TO_EXCITATORY_DELAY_RANGE[] = {time_step, 5.0f*pow(10, -3)};

	//
	simulator.AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INPUT_TO_EXCITATORY_WEIGHT_RANGE, INPUT_TO_EXCITATORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	
	// simulator.AddSynapseGroup(temp_poisson_input_layer, EXCITATORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, false, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);


	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_3, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_4, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);

	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, INHIBITORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, false, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, INHIBITORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, false, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, INHIBITORY_NEURONS_LAYER_3, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_4, INHIBITORY_NEURONS_LAYER_4, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);

	simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, false, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	// simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_4, EXCITATORY_NEURONS_LAYER_4, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, true, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);

	simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_1, INHIBITORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, false, synapse_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);

	clock_t adding_synapses_end = clock();
	float adding_synapses_total_time = float(adding_synapses_end - adding_synapses_start) / CLOCKS_PER_SEC;
	printf("Synapses Added. Time taken: %f\n", adding_synapses_total_time);
	print_line_of_dashes_with_blank_lines_either_side();

	//
	int temp_model_type = 1;
	simulator.setup_network(temp_model_type);
	simulator.setup_recording_electrodes();

	// TRAINING
	float total_time_per_epoch = 3.0f;
	int number_of_epochs = 1;
	bool save_spikes = true;
	bool apply_stdp_to_relevant_synapses = true;
	simulator.Run(total_time_per_epoch, number_of_epochs, temp_model_type, save_spikes, apply_stdp_to_relevant_synapses);

	// TESTING
	// total_time_per_epoch = 1.0f;
	// number_of_epochs = 1;
	// save_spikes = true;
	// apply_stdp_to_relevant_synapses = false;
	// simulator.Run(total_time_per_epoch, number_of_epochs, temp_model_type, save_spikes, apply_stdp_to_relevant_synapses);

	clock_t end_entire_experiment = clock();
	float timed_entire_experiment = float(end_entire_experiment - begin_entire_experiment) / CLOCKS_PER_SEC;
	printf("Entire Experiment Time: %f\n", timed_entire_experiment);


	return 1;
}