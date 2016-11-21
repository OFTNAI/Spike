#ifndef SpikingModel_H
#define SpikingModel_H

#include <stdio.h>
#include "../Synapses/ConductanceSpikingSynapses.hpp"
#include "../STDP/STDP.hpp"
#include "../STDP/EvansSTDP.hpp"
#include "../STDP/HigginsSTDP.hpp"
#include "../STDP/MasquelierSTDP.hpp"
#include "../Neurons/Neurons.hpp"
#include "../Neurons/SpikingNeurons.hpp"
#include "../Neurons/LIFSpikingNeurons.hpp"
#include "../Neurons/AdExSpikingNeurons.hpp"
#include "../Neurons/ImagePoissonInputSpikingNeurons.hpp"
#include "../Neurons/GeneratorInputSpikingNeurons.hpp"
#include "../SpikeAnalyser/SpikeAnalyser.hpp"
#include "../Helpers/TimerWithMessages.hpp"
#include "../Helpers/RandomStateManager.hpp"
// #include "../Helpers/TerminalHelpers.hpp"
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

  void reset_state();
  void perform_per_timestep_model_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses);

  // virtual void finalise_model();
  // TODO
  // virtual void init_backend(Context* ctx);
  virtual void init_backend(bool high_fidelity_spike_storage);
};

#endif
