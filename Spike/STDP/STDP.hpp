// STDP Class Header
// STDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef STDP_H
#define STDP_H

// Get Synapses & Neurons Class
#include "../Synapses/SpikingSynapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

namespace Backend {
  class STDPCommon {
  public:
  };

  class STDP : public virtual STDPCommon,
               public Generic {
  public:
  };
}

#include "Spike/Backend/Dummy/STDP/STDP.hpp"

// STDP Parameters
struct stdp_parameters_struct {
	stdp_parameters_struct() {}
};


class STDP {
public:
  void* _backend;
  ADD_BACKEND_GETTER(STDP);
  
  virtual void prepare_backend(Context* ctx) = 0;
  virtual void reset_state() = 0;

  // Set STDP Parameters
  virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters) = 0;

  // virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep) = 0;
  virtual void Run_STDP(SpikingNeurons* neurons, float current_time_in_seconds, float timestep) = 0;
};

#endif
