// Higgins STDP Class Header
// HigginsSTDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef HIGGINS_STDP_H
#define HIGGINS_STDP_H

// Get Synapses & Neurons Class
#include "../Synapses/SpikingSynapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"
#include "../STDP/STDP.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

class HigginsSTDP; // forward definition

namespace Backend {
  class HigginsSTDP : public virtual STDP {
  public:
    SPIKE_ADD_BACKEND_FACTORY(HigginsSTDP);

    virtual void apply_ltp_to_synapse_weights(float current_time_in_seconds) = 0;
    virtual void apply_ltd_to_synapse_weights(float current_time_in_seconds) = 0;
  };
}

// STDP Parameters
struct higgins_stdp_parameters_struct : stdp_parameters_struct {
  // STDP Parameters
  float w_max = 60.0;
  float a_minus = -0.015;
  float a_plus = 0.005;
  float tau_minus = 0.025;
  float tau_plus = 0.015;
};


class HigginsSTDP : public STDP {
public:
  ~HigginsSTDP() override;
  SPIKE_ADD_BACKEND_GETSET(HigginsSTDP, STDP);

  struct higgins_stdp_parameters_struct* stdp_params = nullptr;

  void init_backend(Context* ctx = _global_ctx) override;

  // Set STDP Parameters
  void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters) override;
  // STDP
  void Run_STDP(float current_time_in_seconds, float timestep) override;

  // LTP & LTD for this model
  void apply_ltd_to_synapse_weights(float current_time_in_seconds);
  void apply_ltp_to_synapse_weights(float current_time_in_seconds);

private:
  std::shared_ptr<::Backend::HigginsSTDP> _backend;
};

#endif
