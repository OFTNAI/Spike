#ifndef SpikingModel_H
#define SpikingModel_H

#define SILENCE_MODEL_SETUP


#include <cuda.h>
#include <stdio.h>
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../STDP/STDP.h"
#include "../STDP/EvansSTDP.h"
#include "../STDP/HigginsSTDP.h"
#include "../STDP/MasquelierSTDP.h"
#include "../STDP/vanRossumSTDP.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/AdExSpikingNeurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"
#include "../Neurons/GeneratorInputSpikingNeurons.h"
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
	bool high_fidelity_spike_storage;
	void SetTimestep(float timestep_parameter);

	SpikingNeurons * spiking_neurons;
	SpikingSynapses * spiking_synapses;
	InputSpikingNeurons * input_spiking_neurons;
	STDP* stdp_rule;

	int AddNeuronGroup(neuron_parameters_struct * group_params);
	int AddInputNeuronGroup(neuron_parameters_struct * group_params);

	void AddSynapseGroup(int presynaptic_group_id, int postsynaptic_group_id, synapse_parameters_struct * synapse_params);
	void AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, synapse_parameters_struct * synapse_params);

	void reset_model_activities();
	void perform_per_timestep_model_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses);

	virtual void finalise_model();
	void copy_model_to_device();

protected:
	
	virtual void create_parameter_arrays();

};

#endif
