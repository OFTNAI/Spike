#ifndef SpikingModel_H
#define SpikingModel_H

#define SILENCE_MODEL_SETUP
class SpikingModel; // Forward Declaration

#include <stdio.h>
#include "../Backend/Context.hpp"
#include "../Synapses/SpikingSynapses.hpp"
#include "../Plasticity/STDPPlasticity.hpp"
#include "../Neurons/Neurons.hpp"
#include "../Neurons/SpikingNeurons.hpp"
#include "../Helpers/TimerWithMessages.hpp"
#include "../Helpers/RandomStateManager.hpp"
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

  Context* context = nullptr; // Call init_backend to set this up!

  float timestep;
  int timestep_grouping = 1;
  bool plasticity_on = false;
  void SetTimestep(float timestep_parameter);

  SpikingNeurons * spiking_neurons = nullptr;
  SpikingSynapses * spiking_synapses = nullptr;
  SpikingNeurons * input_spiking_neurons = nullptr;
  vector<STDPPlasticity*> plasticity_rule_vec; 

  int AddNeuronGroup(neuron_parameters_struct * group_params);
  int AddInputNeuronGroup(neuron_parameters_struct * group_params);
	
  void AddSynapseGroup(int presynaptic_group_id, int postsynaptic_group_id, synapse_parameters_struct * synapse_params);
  void AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, synapse_parameters_struct * synapse_params);

  void AddPlasticityRule(STDPPlasticity * plasticity_rule);
  void ActivatePlasticity(bool apply_plasticity_to_relevant_synapses);

  void reset_state();
  void perform_per_timestep_model_instructions(float current_time_in_seconds);

  virtual void finalise_model();

  virtual void init_backend();
  virtual void prepare_backend();

protected:
  virtual void create_parameter_arrays() {}
};

#endif
