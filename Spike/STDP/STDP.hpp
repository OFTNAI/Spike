// STDP Class Header
// STDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef STDP_H
#define STDP_H

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

class STDP; // forward definition

namespace Backend {
  class STDP : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(STDP);
    ~STDP() override = default;
  };
}

static_assert(std::has_virtual_destructor<Backend::STDP>::value,
              "contract violated");

// STDP Parameters
struct stdp_parameters_struct {
	stdp_parameters_struct() {}
};


class STDP : public virtual SpikeBase {
public:
  ~STDP() override = default;

  SPIKE_ADD_BACKEND_GETSET(STDP, SpikeBase);
  void reset_state() override;

  SpikingSynapses* syns = nullptr;
  SpikingNeurons* neurs = nullptr;
  int stdp_rule_id = -1;

  // Set STDP Parameters
  // TODO: Shouldn't this be done in the constructor?
  virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters) = 0;

  virtual void Run_STDP(float current_time_in_seconds, float timestep) = 0;

private:
  std::shared_ptr<::Backend::STDP> _backend;
};

#endif
