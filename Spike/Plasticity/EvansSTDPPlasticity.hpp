// Evans STDP Class Header
// EvansSTDPPlasticity.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef Evans_STDP_H
#define Evans_STDP_H

// Get Synapses Class
#include "../Synapses/Synapses.hpp"
#include "../Plasticity/STDPPlasticity.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

class EvansSTDPPlasticity; // forward definition

namespace Backend {
  class EvansSTDPPlasticity : public virtual STDPPlasticity {
  public:
    SPIKE_ADD_BACKEND_FACTORY(EvansSTDPPlasticity);

    virtual void update_synaptic_efficacies_or_weights(float current_time_in_seconds) = 0;
    virtual void update_presynaptic_activities(float timestep, float current_time_in_seconds) = 0;
    virtual void update_postsynaptic_activities(float timestep, float current_time_in_seconds) = 0;
  };
}

// STDPPlasticity Parameters
struct evans_stdp_plasticity_parameters_struct : stdp_plasticity_parameters_struct {
  // STDPPlasticity Parameters, with default values
  float decay_term_tau_D = 0.005;
  float model_parameter_alpha_D = 0.5;
  float synaptic_neurotransmitter_concentration_alpha_C = 0.5;
  float decay_term_tau_C = 0.004;
  float learning_rate_rho = 0.1;
};


class EvansSTDPPlasticity : public STDPPlasticity {
public:
  // Constructor/Destructor
  EvansSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters);
  ~EvansSTDPPlasticity() override;
  SPIKE_ADD_BACKEND_GETSET(EvansSTDPPlasticity, STDPPlasticity);

  struct evans_stdp_plasticity_parameters_struct* stdp_params = nullptr;

  //(NEURON-WISE)
  float* recent_postsynaptic_activities_D = nullptr;

  //(SYNAPSE-WISE)
  float* recent_presynaptic_activities_C = nullptr;

  void init_backend(Context* ctx = _global_ctx) override;
  void prepare_backend_late() override;

  void state_update(float current_time_in_seconds, float timestep) override;
	
  // Updates for this model
  void update_presynaptic_activities(float timestep, float current_time_in_seconds);
  void update_synaptic_efficacies_or_weights(float current_time_in_seconds);
  void update_postsynaptic_activities(float timestep, float current_time_in_seconds);

private:
  std::shared_ptr<::Backend::EvansSTDPPlasticity> _backend;
};

#endif
