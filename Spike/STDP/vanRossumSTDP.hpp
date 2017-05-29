// vanRossum STDP Class Header
// vanRossumSTDP.h
//
// This STDP learning rule is extracted from the following paper:

//  Rossum, M. C. van, G. Q. Bi, and G. G. Turrigiano. 2000. “Stable Hebbian Learning from Spike Timing-Dependent Plasticity.” The Journal of Neuroscience: The Official Journal of the Society for Neuroscience 20 (23): 8812–21.

// This equation is based upon the multiplicative learning rule without the gaussian random variable
// The default parameters are also those used in the above paper
//  Author: Nasir Ahmad
//  Date: 03/10/2016

#ifndef VANROSSUM_STDP_H
#define VANROSSUM_STDP_H

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

class vanRossumSTDP; // forward definition

namespace Backend {
  class vanRossumSTDP : public virtual STDP {
  public:
    SPIKE_ADD_BACKEND_FACTORY(vanRossumSTDP);

    virtual void apply_stdp_to_synapse_weights(float current_time_in_seconds) = 0;
  };
}

// STDP Parameters
struct vanrossum_stdp_parameters_struct : stdp_parameters_struct {
  vanrossum_stdp_parameters_struct() : a_minus(0.003), a_plus(7.0f*pow(10.0, -12)), tau_minus(0.02f), tau_plus(0.02f), weight_dependency_factor(1.0f), allspikes(true), timestep(0.0f) { } // default Constructor
  // STDP Parameters
  float a_minus;
  float a_plus;
  float tau_minus;
  float tau_plus;
  float weight_dependency_factor;
  // All-To-All vs Nearest
  bool allspikes;
  float timestep;
};


class vanRossumSTDP : public STDP {
public:
  ~vanRossumSTDP() override;
  SPIKE_ADD_BACKEND_GETSET(vanRossumSTDP, STDP);

  struct vanrossum_stdp_parameters_struct* stdp_params;

  // Nearest STDP Variables
  int* index_of_last_afferent_synapse_to_spike = nullptr;
  bool* isindexed_ltd_synapse_spike = nullptr;
  int* index_of_first_synapse_spiked_after_postneuron = nullptr;

  void init_backend(Context* ctx = _global_ctx) override;
  void prepare_backend_late() override;

  // Set STDP Parameters
  void Set_STDP_Parameters(SpikingSynapses* synapses,
                           SpikingNeurons* neurons,
                           SpikingNeurons* input_neurons,
                           stdp_parameters_struct* stdp_parameters) override;
  // STDP
  void Run_STDP(float current_time_in_seconds, float timestep) override;

  // LTP & LTD for this model
  void apply_stdp_to_synapse_weights(float current_time_in_seconds);

private:
  std::shared_ptr<::Backend::vanRossumSTDP> _backend;
};

#endif
