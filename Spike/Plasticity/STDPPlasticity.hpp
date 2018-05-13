// STDPPlasticity Class Header
// STDPPlasticity.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef STDP_H
#define STDP_H

class STDPPlasticity; // forward definition

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Plasticity/Plasticity.hpp"
#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Models/SpikingModel.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>
#include <vector>

namespace Backend {
  class STDPPlasticity : public virtual Plasticity {
  public:
    SPIKE_ADD_BACKEND_FACTORY(STDPPlasticity);
    ~STDPPlasticity() override = default;
  };
}

static_assert(std::has_virtual_destructor<Backend::STDPPlasticity>::value,
              "contract violated");

// STDPPlasticity Parameters
struct stdp_plasticity_parameters_struct : plasticity_parameters_struct {
	stdp_plasticity_parameters_struct() {}
};


class STDPPlasticity : public Plasticity {
public:
  ~STDPPlasticity() override;

  SPIKE_ADD_BACKEND_GETSET(STDPPlasticity, SpikeBase);
  void reset_state() override;

  SpikingSynapses* syns = nullptr;
  SpikingNeurons* neurs = nullptr;
  SpikingModel* model = nullptr;
  
  // Storage locations for the neurons and synapses involved in plasticity
  // Only axonal transmission delays are used
  // Dealt with by AddSynapse function
  std::vector<int> plastic_synapses;
  // Details on post-synaptic neuron synapse relations
  std::vector<int> post_neuron_set;
  std::vector<int> post_neuron_conversion;
  // Details on pre-synaptic neuron synapse relations
  std::vector<int> pre_neuron_set;
  std::vector<int> pre_neuron_conversion;
  std::vector<int> pre_input_neuron_conversion;
  std::vector<int> pre_neuron_efferent_counts;
  std::vector<int*> pre_neuron_efferent_ids;

  virtual void AddSynapse(int presynaptic_neuron, int postsynaptic_neuron, int synapse_id);
  virtual void state_update(float current_time_in_seconds, float timestep) override = 0;

private:
  std::shared_ptr<::Backend::STDPPlasticity> _backend;
};

#endif
