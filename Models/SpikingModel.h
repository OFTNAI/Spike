#ifndef SpikingModel_H
#define SpikingModel_H

#include <cuda.h>
#include <stdio.h>
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../STDP/STDP.h"
#include "../STDP/EvansSTDP.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"
// #include "../Helpers/TerminalHelpers.h"
#include <string>
#include <fstream>
#include <vector>

#include <iostream>
using namespace std;


class SpikingModel {

public:

	// Constructor/Destructor
	SpikingModel();
	~SpikingModel();


	float timestep;
	void SetTimestep(float timestep_parameter);

	SpikingNeurons * spiking_neurons;
	SpikingSynapses * spiking_synapses;
	InputSpikingNeurons * input_spiking_neurons;
	STDP* stdp_rule; 


	int AddNeuronGroup(neuron_parameters_struct * group_params);
	int AddInputNeuronGroup(neuron_parameters_struct * group_params);
	
	void AddSynapseGroup(int presynaptic_group_id, int postsynaptic_group_id, synapse_parameters_struct * synapse_params);
	void AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, synapse_parameters_struct * synapse_params);

	void step_1();
	void step_2(bool is_optimisation);
	void copy_model_to_device(bool high_fidelity_spike_storage);

};

#endif