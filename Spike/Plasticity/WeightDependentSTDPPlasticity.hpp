#ifndef WEIGHTDEPENDENT_STDP_H
#define WEIGHTDEPENDENT_STDP_H

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

class WeightDependentSTDPPlasticity; // forward definition

namespace Backend {
  class WeightDependentSTDPPlasticity : public virtual STDPPlasticity {
  public:
    SPIKE_ADD_BACKEND_FACTORY(WeightDependentSTDPPlasticity);

    virtual void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) = 0;
  };
}

// STDP Parameters
struct weightdependent_stdp_plasticity_parameters_struct : stdp_plasticity_parameters_struct {
  weightdependent_stdp_plasticity_parameters_struct() : 
    a_minus(1.0), a_plus(1.0), tau_minus(0.02f), tau_plus(0.02f), lambda(1.0f), alpha(1.0f), w_max(1.0f), nearest_spike_only(false) { } // default Constructor
  // STDPPlasticity Parameters
  float a_minus;
  float a_plus;
  float tau_minus;
  float tau_plus;
  float lambda;
  float alpha;
  float w_max;
  bool nearest_spike_only;
};


class WeightDependentSTDPPlasticity : public STDPPlasticity {
public:
  WeightDependentSTDPPlasticity(SpikingSynapses* synapses,
                           SpikingNeurons* neurons,
                           SpikingNeurons* input_neurons,
                           stdp_plasticity_parameters_struct* stdp_parameters);
  ~WeightDependentSTDPPlasticity() override;
  SPIKE_ADD_BACKEND_GETSET(WeightDependentSTDPPlasticity, STDPPlasticity);

  struct weightdependent_stdp_plasticity_parameters_struct* stdp_params;

  void init_backend(Context* ctx = _global_ctx) override;
  //void prepare_backend_late() override;

  void state_update(float current_time_in_seconds, float timestep) override;

  void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep);

private:
  std::shared_ptr<::Backend::WeightDependentSTDPPlasticity> _backend;
};

#endif
