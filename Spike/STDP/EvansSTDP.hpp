// Evans STDP Class Header
// EvansSTDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef Evans_STDP_H
#define Evans_STDP_H

// Get Synapses Class
#include "../Synapses/Synapses.hpp"
#include "../STDP/STDP.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

class EvansSTDP; // forward definition

namespace Backend {
  class EvansSTDP : public virtual STDP {
  public:
    ADD_FRONTEND_GETTER(EvansSTDP);

    void prepare() override {
      printf("TODO Backend::EvansSTDP::prepare\n");
    }

    void push_data_front() override {} // TODO
    void pull_data_back() override {} // TODO

    virtual void update_synaptic_efficacies_or_weights(float current_time_in_seconds) = 0;
    virtual void update_presynaptic_activities(float timestep, float current_time_in_seconds) = 0;
    virtual void update_postsynaptic_activities(float timestep, float current_time_in_seconds) = 0;
  };
}

#include "Spike/Backend/Dummy/STDP/EvansSTDP.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/STDP/EvansSTDP.hpp"
#endif

// STDP Parameters
struct evans_stdp_parameters_struct : stdp_parameters_struct {
  // STDP Parameters, with default values
  float decay_term_tau_D = 0.005;
  float model_parameter_alpha_D = 0.5;
  float synaptic_neurotransmitter_concentration_alpha_C = 0.5;
  float decay_term_tau_C = 0.004;
  float learning_rate_rho = 0.1;
};


class EvansSTDP : public virtual STDP {
public:
  // Constructor/Destructor
  ~EvansSTDP();
  ADD_BACKEND_GETTER(EvansSTDP);

  struct evans_stdp_parameters_struct* stdp_params = nullptr;

  //(NEURON-WISE)
  float* recent_postsynaptic_activities_D = nullptr;

  //(SYNAPSE-WISE)
  float* recent_presynaptic_activities_C = nullptr;

  void prepare_backend(Context* ctx = _global_ctx) override;
  inline void prepare_backend_extra();
  void reset_state() override;

  // Set STDP Parameters
  void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters) override;
	
  void Run_STDP(float current_time_in_seconds, float timestep) override;
	
  // Updates for this model
  void update_presynaptic_activities(float timestep, float current_time_in_seconds);
  void update_synaptic_efficacies_or_weights(float current_time_in_seconds);
  void update_postsynaptic_activities(float timestep, float current_time_in_seconds);
};

#endif
