// Masquelier STDP Class Header
// MasquelierSTDP.h
//
// This STDP learning rule is extracted from the following paper:

//	Timothee Masquelier, Rudy Guyonneau, and Simon J Thorpe. Spike timing 
//	dependent plasticity finds the start of repeating patterns in continuous spike
//	trains. PLoS One, 3(1):e1377, 2 January 2008.


// The default parameters are also those used in the above paper
//	Author: Nasir Ahmad
//	Date: 03/10/2016

#ifndef MASQUELIER_STDP_H
#define MASQUELIER_STDP_H

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

class MasquelierSTDP; // forward definition

namespace Backend {
  class MasquelierSTDP : public virtual STDP {
  public:
    SPIKE_ADD_BACKEND_FACTORY(MasquelierSTDP);

    virtual void apply_stdp_to_synapse_weights(float current_time_in_seconds) = 0;
  };
}

// STDP Parameters
struct masquelier_stdp_parameters_struct : stdp_parameters_struct {
  // STDP Parameters
  float a_minus = 0.85*0.03125;
  float a_plus = 0.03125;
  float tau_minus = 0.033;
  float tau_plus = 0.0168;
};


class MasquelierSTDP : public STDP {
public:
  ~MasquelierSTDP() override;
  SPIKE_ADD_BACKEND_GETSET(MasquelierSTDP, STDP);

  struct masquelier_stdp_parameters_struct* stdp_params = nullptr;

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
  std::shared_ptr<::Backend::MasquelierSTDP> _backend;
};

#endif
