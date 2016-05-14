// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE="Example" model


#include "../Simulator/Simulator.h"
#include "../Synapses/IzhikevichSpikingSynapses.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/IzhikevichSpikingNeurons.h"
#include "../Neurons/PoissonSpikingNeurons.h"

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	// Set the time_stepep of the simulation as required (time_stepep is measure in seconds)
	
	// Create an instance of the Simulator and set the time_step
	Simulator simulator;
	float time_step = 0.001;
	simulator.SetTimestep(time_step);
	simulator.SetNeuronType(new IzhikevichSpikingNeurons());
	simulator.SetInputNeuronType(new PoissonSpikingNeurons());
	simulator.SetSynapseType(new IzhikevichSpikingSynapses());

	//
	poisson_spiking_neuron_parameters_struct * poisson_spiking_group_params = new poisson_spiking_neuron_parameters_struct();
	poisson_spiking_group_params->rate = 30.0f;

	//
	izhikevich_spiking_neuron_parameters_struct * izhhikevich_spiking_group_params = new izhikevich_spiking_neuron_parameters_struct();
	izhhikevich_spiking_group_params->parama = 0.02f;
	izhhikevich_spiking_group_params->paramb = -0.01f;
	izhhikevich_spiking_group_params->after_spike_reset_membrane_potential_c = -55.0f;
	izhhikevich_spiking_group_params->paramd = 6.0f;

	//
	int LAYER_1_SHAPE[] = {100, 100};
	int LAYER_2_SHAPE[] = {101, 10};
	int LAYER_3_SHAPE[] = {10, 10};
	// int LAYER_1_SHAPE[] = {5, 5};
	// int LAYER_2_SHAPE[] = {5, 5};
	// int LAYER_3_SHAPE[] = {5, 5};

	//
	int POISSON_SPIKING_GROUP_ID_LAYER_1 = simulator.AddInputNeuronGroup(poisson_spiking_group_params, LAYER_1_SHAPE);
	int IZHIKEVICH_SPIKING_GROUP_ID_LAYER_2 = simulator.AddNeuronGroup(izhhikevich_spiking_group_params, LAYER_2_SHAPE);
	int IZHIKEVICH_SPIKING_GROUP_ID_LAYER_3 = simulator.AddNeuronGroup(izhhikevich_spiking_group_params, LAYER_3_SHAPE);
	
	//
	// float LAYER_1_TO_LAYER_2_WEIGHTS[] = {2.50f, 5.0f};
	// float LAYER_2_TO_LAYER_3_WEIGHTS[] = {2.50f, 5.0f};
	float LAYER_1_TO_LAYER_2_WEIGHTS[] = {0.02, 0.20f};
	float LAYER_2_TO_LAYER_3_WEIGHTS[] = {0.02, 0.20f};

	//
	float LAYER_1_TO_LAYER_2_DELAY_RANGE[] = {time_step, time_step};
	float LAYER_2_TO_LAYER_3_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};

	//
	simulator.AddSynapseGroup(POISSON_SPIKING_GROUP_ID_LAYER_1, IZHIKEVICH_SPIKING_GROUP_ID_LAYER_2, CONNECTIVITY_TYPE_ALL_TO_ALL, LAYER_1_TO_LAYER_2_WEIGHTS, LAYER_1_TO_LAYER_2_DELAY_RANGE, false);
	simulator.AddSynapseGroup(IZHIKEVICH_SPIKING_GROUP_ID_LAYER_2, IZHIKEVICH_SPIKING_GROUP_ID_LAYER_3, CONNECTIVITY_TYPE_ALL_TO_ALL, LAYER_2_TO_LAYER_3_WEIGHTS, LAYER_2_TO_LAYER_3_DELAY_RANGE, true);

	//
	float total_time_per_epoch = 5.0f;
	int number_of_epochs = 5;
	bool save_spikes = true;

	int temp_model_type = 0;
	simulator.initialise_network(temp_model_type);
	simulator.initialise_recording_electrodes();

	//
	simulator.Run(total_time_per_epoch, number_of_epochs, temp_model_type, save_spikes);


	return 1;
}