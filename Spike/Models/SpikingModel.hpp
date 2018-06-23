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
#include "../ActivityMonitor/ActivityMonitor.hpp"
#include <string>
#include <fstream>
#include <vector>

#include <iostream>
using namespace std;


class SpikingModel {
private:
  void perform_per_step_model_instructions();
  virtual void finalise_model();
public:
  // Constructor/Destructor
  //SpikingModel(SpikingNeurons* spiking_neurons, SpikingNeurons* input_spiking_neurons, SpikingSynapses* spiking_synapses);
  SpikingModel();
  ~SpikingModel();

  Context* context = nullptr; // Call init_backend to set this up!
  SpikingNeurons * spiking_neurons = nullptr;
  SpikingSynapses * spiking_synapses = nullptr;
  SpikingNeurons * input_spiking_neurons = nullptr;
  
  vector<STDPPlasticity*> plasticity_rule_vec; 
  vector<ActivityMonitor*> monitors_vec; 

  bool model_complete = false;

  float timestep;
  int current_time_in_timesteps = 0;
  int timestep_grouping = 1;
  void SetTimestep(float timestep_parameter);

  int AddNeuronGroup(neuron_parameters_struct * group_params);
  int AddInputNeuronGroup(neuron_parameters_struct * group_params);

  void AddSynapseGroup(int presynaptic_group_id, int postsynaptic_group_id, synapse_parameters_struct * synapse_params);
  void AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, synapse_parameters_struct * synapse_params);

  void AddPlasticityRule(STDPPlasticity * plasticity_rule);
  void AddActivityMonitor(ActivityMonitor * activityMonitor);

  void reset_state();
  void run(float seconds);

  virtual void init_backend();
  virtual void prepare_backend();

protected:
  virtual void create_parameter_arrays() {}
};

#endif
