// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE="ExampleLIF" model


#include "Simulator/Simulator.h"
#include "Synapses/LIFSpikingSynapses.h"
#include "Neurons/Neurons.h"
#include "Neurons/LIFSpikingNeurons.h"
#include "Neurons/PoissonSpikingNeurons.h"

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	// Set the time_stepep of the simulation as required (time_stepep is measure in seconds)
	
	// Create an instance of the Simulator and set the time_step
	Simulator simulator;
	float time_step = 0.001;
	simulator.SetTimestep(time_step);
	simulator.SetNeuronType(new LIFSpikingNeurons());
	simulator.SetInputNeuronType(new PoissonSpikingNeurons());
	simulator.SetSynapseType(new LIFSpikingSynapses());

	//
	poisson_spiking_neuron_parameters_struct * poisson_spiking_group_params = new poisson_spiking_neuron_parameters_struct();
	poisson_spiking_group_params->rate = 30.0f;

	//
	lif_spiking_neuron_parameters_struct * lif_spiking_group_params = new lif_spiking_neuron_parameters_struct();
	lif_spiking_group_params->after_spike_reset_membrane_potential_c = -55.0f;
	lif_spiking_group_params->paramd = 6.0f; //Old Izhikevich parameter. Leaving temporarily so spikes

	//
	int LAYER_1_SHAPE[] = {100, 100};
	int LAYER_2_SHAPE[] = {101, 10};
	int LAYER_3_SHAPE[] = {10, 10};

	//
	int POISSON_SPIKING_GROUP_ID_LAYER_1 = simulator.AddInputNeuronGroup(poisson_spiking_group_params, LAYER_1_SHAPE);
	int LIF_SPIKING_GROUP_ID_LAYER_2 = simulator.AddNeuronGroup(lif_spiking_group_params, LAYER_2_SHAPE);
	int LIF_SPIKING_GROUP_ID_LAYER_3 = simulator.AddNeuronGroup(lif_spiking_group_params, LAYER_3_SHAPE);
	
	//
	// float LAYER_1_TO_LAYER_2_WEIGHTS[] = {2.50f, 5.0f};
	// float LAYER_2_TO_LAYER_3_WEIGHTS[] = {2.50f, 5.0f};
	float LAYER_1_TO_LAYER_2_WEIGHTS[] = {0.0, 18.0f*pow(10, -9)};
	float LAYER_2_TO_LAYER_3_WEIGHTS[] = {0.0, 18.0f*pow(10, -9)};

	//
	float LAYER_1_TO_LAYER_2_DELAY_RANGE[] = {time_step, time_step};
	float LAYER_2_TO_LAYER_3_DELAY_RANGE[] = {time_step, 50.0f*pow(10, -3)};

	//
	simulator.AddSynapseGroup(POISSON_SPIKING_GROUP_ID_LAYER_1, LIF_SPIKING_GROUP_ID_LAYER_2, CONNECTIVITY_TYPE_ALL_TO_ALL, LAYER_1_TO_LAYER_2_WEIGHTS, LAYER_1_TO_LAYER_2_DELAY_RANGE, false);
	simulator.AddSynapseGroup(LIF_SPIKING_GROUP_ID_LAYER_2, LIF_SPIKING_GROUP_ID_LAYER_3, CONNECTIVITY_TYPE_ALL_TO_ALL, LAYER_2_TO_LAYER_3_WEIGHTS, LAYER_2_TO_LAYER_3_DELAY_RANGE, true);

	//
	float total_time_per_epoch = 1.25f;
	int number_of_epochs = 1;
	bool save_spikes = true;

	//
	simulator.Run(total_time_per_epoch, number_of_epochs, save_spikes, 1);


	return 1;
}