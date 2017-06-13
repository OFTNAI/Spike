// Plasticity Class Header
// Plasticity.h
//
//	Author: Nasir Ahmad
//	Date: 29/05/2017

#ifndef PLASTICITY_H
#define PLASTICITY_H

class Plasticity; // forward definition

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Synapses/Synapses.hpp"
//#include "Spike/Neurons/SpikingNeurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

namespace Backend {
  class Plasticity : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(Plasticity);
    ~Plasticity() override = default;
  };
}

static_assert(std::has_virtual_destructor<Backend::Plasticity>::value,
              "contract violated");

// Plasticity Parameters
struct plasticity_parameters_struct {
	plasticity_parameters_struct() {}
};


class Plasticity : public virtual SpikeBase {
public:
  ~Plasticity() override = default;

  SPIKE_ADD_BACKEND_GETSET(Plasticity, SpikeBase);
  void reset_state() override;

  // ID for Plasticity Rules
  int plasticity_rule_id = -1;

  virtual void state_update(float current_time_in_seconds, float timestep) = 0;

private:
  std::shared_ptr<::Backend::Plasticity> _backend;
};

#endif
