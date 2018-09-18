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
  SpikingNeurons* in_neurs = nullptr;
  SpikingNeurons* neurs = nullptr;
  SpikingModel* model = nullptr;
  
  // Dealt with by AddSynapseIndices function
  std::vector<int> plastic_synapses;
  int total_number_of_plastic_synapses = 0;

  virtual void AddSynapseIndices(int synapse_id_start, int num_synapses_to_add);
  virtual void state_update(float current_time_in_seconds, float timestep) override = 0;

private:
  std::shared_ptr<::Backend::STDPPlasticity> _backend;
};

#endif
