// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE="Experiments/ConductanceExperiment1" model


#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/ConductanceSpikingNeurons.h"
#include "../Neurons/ImagePoissonSpikingNeurons.h"
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
	simulator.SetInputNeuronType(new ImagePoissonSpikingNeurons());
	simulator.SetSynapseType(new ConductanceSpikingSynapses());

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
	printf("Input Neurons Added. Time Taken: %f\n\n", adding_input_neurons_total_time);


	/////////// ADD NEURONS ///////////
	printf("Adding Neurons...\n");
	clock_t adding_neurons_start = clock();

	//
	conductance_spiking_neuron_parameters_struct * conductance_spiking_group_params = new conductance_spiking_neuron_parameters_struct();
	conductance_spiking_group_params->after_spike_reset_membrane_potential_c = -74.0f;
	conductance_spiking_group_params->threshold_for_action_potential_spike = -53.0f;
	conductance_spiking_group_params->paramd = 6.0f; //Old Izhikevich parameter. Leaving temporarily so spikes

	//
	int EXCITATORY_LAYER_SHAPE[] = {64, 64};
	int INHIBITORY_LAYER_SHAPE[] = {16, 16};

	int EXCITATORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(conductance_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int EXCITATORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(conductance_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int EXCITATORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(conductance_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int EXCITATORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(conductance_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(conductance_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(conductance_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(conductance_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(conductance_spiking_group_params, INHIBITORY_LAYER_SHAPE);

	clock_t adding_neurons_end = clock();
	float adding_neurons_total_time = float(adding_neurons_end - adding_neurons_start) / CLOCKS_PER_SEC;
	printf("Neurons Added. Time taken: %f\n\n", adding_neurons_total_time);


	/////////// ADD SYNAPSES ///////////
	printf("Adding Synapses...\n");
	clock_t adding_synapses_start = clock();

	float CONNECTIVITY_STANDARD_DEVIATION_SIGMA = 5.0;

	connectivity_parameters_struct * connectivity_parameters = new connectivity_parameters_struct();
	connectivity_parameters->max_number_of_connections_per_pair = 5;

	//
	float INPUT_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -7)};
	float EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -9)};
	float EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -9)};
	float INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -9)};

	//
	float INPUT_TO_EXCITATORY_DELAY_RANGE[] = {time_step, time_step};
	float EXCITATORY_TO_EXCITATORY_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};
	float EXCITATORY_TO_INHIBITORY_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};
	float INHIBITORY_TO_EXCITATORY_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};

	//
	simulator.AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INPUT_TO_EXCITATORY_WEIGHT_RANGE, INPUT_TO_EXCITATORY_DELAY_RANGE, false, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	
	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_3, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_4, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE, EXCITATORY_TO_EXCITATORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);

	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_1, INHIBITORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, false, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_2, INHIBITORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, false, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_3, INHIBITORY_NEURONS_LAYER_3, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(EXCITATORY_NEURONS_LAYER_4, INHIBITORY_NEURONS_LAYER_4, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE, EXCITATORY_TO_INHIBITORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);

	simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_1, EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_2, EXCITATORY_NEURONS_LAYER_2, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_3, EXCITATORY_NEURONS_LAYER_3, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(INHIBITORY_NEURONS_LAYER_4, EXCITATORY_NEURONS_LAYER_4, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE, INHIBITORY_TO_EXCITATORY_DELAY_RANGE, true, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);


	clock_t adding_synapses_end = clock();
	float adding_synapses_total_time = float(adding_synapses_end - adding_synapses_start) / CLOCKS_PER_SEC;
	printf("Synapses Added. Time taken: %f\n\n", adding_synapses_total_time);

	//
	int temp_model_type = 1;
	simulator.initialise_network(temp_model_type);
	simulator.initialise_recording_electrodes();

	//
	float total_time_per_epoch = 1.0f;
	int number_of_epochs = 1;
	bool save_spikes = true;
	simulator.Run(total_time_per_epoch, number_of_epochs, temp_model_type, save_spikes);

	clock_t end_entire_experiment = clock();
	float timed_entire_experiment = float(end_entire_experiment - begin_entire_experiment) / CLOCKS_PER_SEC;
	printf("Entire Experiment Time: %f\n", timed_entire_experiment);


	return 1;
}