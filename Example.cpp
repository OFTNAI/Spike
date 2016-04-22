// An Example Model for running the SPIKE simulator
//
//	Author: Nasir Ahmad
//	Date: 16/03/2016

//  Adapted by Nasir Ahmad and James Isbister
//	Date: 23/3/2016

// To create the executable for this network, run:
// make FILE="Example" model


#include "Simulator.h"
#include "Constants.h"
#include "Neurons.h"
#include "IzhikevichSpikingNeurons.h"
#include "PoissonSpikingNeurons.h"
// #include "Inputs.h"
// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	// Set the timestep of the simulation as required (timestep is measure in seconds)
	float timest = 0.001;
	// Create an instance of the Simulator and set the timestep
	
	Simulator simulator2;
	simulator2.SetTimestep(timest);
	simulator2.SetNeuronType(new IzhikevichSpikingNeurons());
	simulator2.SetInputNeuronType(new PoissonSpikingNeurons());

	izhikevich_spiking_neuron_parameters_struct * izhhikevich_spiking_group_params = new izhikevich_spiking_neuron_parameters_struct();
	// izhhikevich_spiking_group_params->parama = 0.02f;
	// izhhikevich_spiking_group_params->paramb = -0.01f;
	// izhhikevich_spiking_group_params->paramc = -55.0f;
	// izhhikevich_spiking_group_params->paramd = 6.0f;
	izhhikevich_spiking_group_params->parama = 0.01f;
	izhhikevich_spiking_group_params->paramb = 0.2f;
	izhhikevich_spiking_group_params->paramc = -65.0f;
	izhhikevich_spiking_group_params->paramd = 8.0f;

	poisson_spiking_neuron_parameters_struct * poisson_spiking_group_params = new poisson_spiking_neuron_parameters_struct();
	poisson_spiking_group_params->rate = 30.0f;

	int ji_test_shape[] = {10, 1};
	int ji_test_shape2[] = {20, 1};

	int POISSON_SPIKING_GROUP_ID_LAYER_1 = simulator2.AddInputNeuronGroup(poisson_spiking_group_params, ji_test_shape);
	// int POISSON_SPIKING_GROUP_ID_LAYER_1b = simulator2.AddInputNeuronGroup(poisson_spiking_group_params, ji_test_shape);
	int IZHIKEVICH_SPIKING_GROUP_ID_LAYER_2 = simulator2.AddNeuronGroup(izhhikevich_spiking_group_params, ji_test_shape2);
	// int IZHIKEVICH_SPIKING_GROUP_ID_LAYER_2b = simulator2.AddNeuronGroup(izhhikevich_spiking_group_params, ji_test_shape);
	

	float LAYER_1_TO_LAYER_2_WEIGHTS[] = {2.50f, 5.0f};
	float LAYER_1_TO_LAYER_2_DELAY_RANGE[] = {timest, timest};
	simulator2.AddConnectionGroup(POISSON_SPIKING_GROUP_ID_LAYER_1, IZHIKEVICH_SPIKING_GROUP_ID_LAYER_2, CONNECTIVITY_TYPE_ALL_TO_ALL, LAYER_1_TO_LAYER_2_WEIGHTS, LAYER_1_TO_LAYER_2_DELAY_RANGE, false);
	// simulator2.AddConnectionGroup(POISSON_SPIKING_GROUP_ID_LAYER_1b, IZHIKEVICH_SPIKING_GROUP_ID_LAYER_2b, CONNECTIVITY_TYPE_ALL_TO_ALL, LAYER_1_TO_LAYER_2_WEIGHTS, LAYER_1_TO_LAYER_2_DELAY_RANGE, false);

		// Run the actual simulator!
	float total_time_per_epoch = 1.25f;
	int number_of_epochs = 1;
	bool save_spikes = true;
	simulator2.Run(total_time_per_epoch, number_of_epochs, save_spikes);


	// // connections parameters
	// 
	// 

	// float test_weight_range[] = {2.50f, 5.0f};
	// float test_delay_range[] = {timest, 50.0f*pow(10, -3)};
	// bool stdp_on = true;

	// //Add Connection Groups
	// simulator.AddConnectionGroup(INPUTLAYER, test_neuron_group_1_id, CONNECTIVITY_TYPE_ALL_TO_ALL, INPUT_TO_INTER_weights, DefaultDelay, false);
	// simulator.AddConnectionGroup(test_neuron_group_1_id, test_neuron_group_2_id, CONNECTIVITY_TYPE_ALL_TO_ALL, test_weight_range, test_delay_range, stdp_on, 3.0, 8.0);

	return 1;
}