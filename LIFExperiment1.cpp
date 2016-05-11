// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE="LIFExperiment1" model


#include "Simulator/Simulator.h"
#include "Synapses/LIFSpikingSynapses.h"
#include "Neurons/Neurons.h"
#include "Neurons/LIFSpikingNeurons.h"
#include "Neurons/ImagePoissonSpikingNeurons.h"
#include <time.h>

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	// Set the time_stepep of the simulation as required (time_stepep is measure in seconds)
	
	// Create an instance of the Simulator and set the time_step
	Simulator simulator;
	float time_step = 0.0001;
	simulator.SetTimestep(time_step);
	simulator.SetNeuronType(new LIFSpikingNeurons());
	simulator.SetInputNeuronType(new ImagePoissonSpikingNeurons());
	simulator.SetSynapseType(new LIFSpikingSynapses());

	//
	poisson_spiking_neuron_parameters_struct * poisson_spiking_group_params = new poisson_spiking_neuron_parameters_struct();
	poisson_spiking_group_params->rate = 30.0f;

	//
	lif_spiking_neuron_parameters_struct * lif_spiking_group_params = new lif_spiking_neuron_parameters_struct();
	lif_spiking_group_params->after_spike_reset_membrane_potential_c = -74.0f;
	lif_spiking_group_params->threshold_for_action_potential_spike = -53.0f;
	lif_spiking_group_params->paramd = 6.0f; //Old Izhikevich parameter. Leaving temporarily so spikes

	//
	int INPUT_LAYER_SHAPE[] = {64, 64};
	int EXCITATORY_LAYER_SHAPE[] = {64, 64};
	int INHIBITORY_LAYER_SHAPE[] = {32, 32};

	float CONNECTIVITY_STANDARD_DEVIATION_SIGMA = 5.0;

	connectivity_parameters_struct * connectivity_parameters = new connectivity_parameters_struct();
	connectivity_parameters->max_number_of_connections_per_pair = 5;

	//
	int INPUT_NEURONS_GROUP_1 = simulator.AddInputNeuronGroup(poisson_spiking_group_params, INPUT_LAYER_SHAPE);
	int INPUT_NEURONS_GROUP_2 = simulator.AddInputNeuronGroup(poisson_spiking_group_params, INPUT_LAYER_SHAPE);
	int INPUT_NEURONS_GROUP_3 = simulator.AddInputNeuronGroup(poisson_spiking_group_params, INPUT_LAYER_SHAPE);
	int INPUT_NEURONS_GROUP_4 = simulator.AddInputNeuronGroup(poisson_spiking_group_params, INPUT_LAYER_SHAPE);
	int EXCITATORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(lif_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int EXCITATORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(lif_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int EXCITATORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(lif_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int EXCITATORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(lif_spiking_group_params, EXCITATORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_1 = simulator.AddNeuronGroup(lif_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_2 = simulator.AddNeuronGroup(lif_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_3 = simulator.AddNeuronGroup(lif_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	int INHIBITORY_NEURONS_LAYER_4 = simulator.AddNeuronGroup(lif_spiking_group_params, INHIBITORY_LAYER_SHAPE);
	
	//
	float INPUT_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -6)};
	float EXCITATORY_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -9)};
	float EXCITATORY_TO_INHIBITORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -9)};
	float INHIBITORY_TO_EXCITATORY_WEIGHT_RANGE[] = {0.0, 18.0f*pow(10, -9)};

	//
	float INPUT_TO_EXCITATORY_DELAY_RANGE[] = {time_step, time_step};
	float EXCITATORY_TO_EXCITATORY_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};
	float EXCITATORY_TO_INHIBITORY_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};
	float INHIBITORY_TO_EXCITATORY_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};

	printf("Setting up synapses...\n");
	clock_t begin = clock();

	//
	simulator.AddSynapseGroup(INPUT_NEURONS_GROUP_1, EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INPUT_TO_EXCITATORY_WEIGHT_RANGE, INPUT_TO_EXCITATORY_DELAY_RANGE, false, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(INPUT_NEURONS_GROUP_2, EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INPUT_TO_EXCITATORY_WEIGHT_RANGE, INPUT_TO_EXCITATORY_DELAY_RANGE, false, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(INPUT_NEURONS_GROUP_3, EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INPUT_TO_EXCITATORY_WEIGHT_RANGE, INPUT_TO_EXCITATORY_DELAY_RANGE, false, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	simulator.AddSynapseGroup(INPUT_NEURONS_GROUP_4, EXCITATORY_NEURONS_LAYER_1, CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, INPUT_TO_EXCITATORY_WEIGHT_RANGE, INPUT_TO_EXCITATORY_DELAY_RANGE, false, connectivity_parameters, CONNECTIVITY_STANDARD_DEVIATION_SIGMA);
	
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


	clock_t end = clock();
	float timed = float(end-begin) / CLOCKS_PER_SEC;
	printf("Synapses set up! Time Elapsed: %f\n\n", timed);

	//
	float total_time_per_epoch = 1.0f;
	int number_of_epochs = 1;
	bool save_spikes = true;

	//
	simulator.Run(total_time_per_epoch, number_of_epochs, save_spikes, 1);


	return 1;
}