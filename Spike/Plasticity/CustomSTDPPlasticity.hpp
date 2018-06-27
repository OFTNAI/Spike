#ifndef CUSTOM_STDP_H
#define CUSTOM_STDP_H

// Get Synapses & Neurons Class
#include "../Synapses/SpikingSynapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"
#include "../Plasticity/STDPPlasticity.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

class CustomSTDPPlasticity; // forward definition

namespace Backend {
  class CustomSTDPPlasticity : public virtual STDPPlasticity {
  public:
    SPIKE_ADD_BACKEND_FACTORY(CustomSTDPPlasticity);

    virtual void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) = 0;
  };
}

// STDP Parameters
struct custom_stdp_plasticity_parameters_struct : stdp_plasticity_parameters_struct {
  custom_stdp_plasticity_parameters_struct() : 
    a_minus(1.0), a_plus(1.0), tau_minus(0.02f), tau_plus(0.02f), w_max(1.0f), nearest_spike_only(false) { } // default Constructor
  // STDPPlasticity Parameters
  float a_minus;
  float a_plus;
  float tau_minus;
  float tau_plus;
  float w_max;
  bool nearest_spike_only;
};


class CustomSTDPPlasticity : public STDPPlasticity {
public:
  CustomSTDPPlasticity(SpikingSynapses* synapses,
                           SpikingNeurons* neurons,
                           SpikingNeurons* input_neurons,
                           stdp_plasticity_parameters_struct* stdp_parameters);
  ~CustomSTDPPlasticity() override;
  SPIKE_ADD_BACKEND_GETSET(CustomSTDPPlasticity, STDPPlasticity);

  struct custom_stdp_plasticity_parameters_struct* stdp_params;

  void init_backend(Context* ctx = _global_ctx) override;
  void prepare_backend_late() override;

  void state_update(float current_time_in_seconds, float timestep) override;

  void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep);

private:
  std::shared_ptr<::Backend::CustomSTDPPlasticity> _backend;
};

#endif
