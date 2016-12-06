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
    ADD_FRONTEND_GETTER(STDP);
  };
}

#include "Spike/Backend/Dummy/STDP/STDP.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/STDP/STDP.hpp"
#endif


// STDP Parameters
struct stdp_parameters_struct {
	stdp_parameters_struct() {}
};


class STDP : public virtual SpikeBase {
public:
  ADD_BACKEND_GETTER(STDP);

  SpikingSynapses* syns = nullptr;
  SpikingNeurons* neurs = nullptr;

  // Set STDP Parameters
  virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters) = 0;

  virtual void Run_STDP(float current_time_in_seconds, float timestep) = 0;
};

#endif
