// Vogels STDPPlasticity Class Header
// VogelsSTDPPlasticity.h
//
// This STDPPlasticity learning rule is extracted from the following paper:

//  Vogels, T. P., H. Sprekeler, F. Zenke, C. Clopath, and W. Gerstner. 2011. “Inhibitory Plasticity Balances Excitation and Inhibition in Sensory Pathways and Memory Networks.” Science 334 (6062): 1569–73.

// This implementation is based upon the inhibitory STDP rule described for detailed balance


#ifndef VOGELS_STDP_H
#define VOGELS_STDP_H

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

class VogelsSTDPPlasticity; // forward definition

namespace Backend {
  class VogelsSTDPPlasticity : public virtual STDPPlasticity {
  public:
    SPIKE_ADD_BACKEND_FACTORY(VogelsSTDPPlasticity);

    virtual void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) = 0;
  };
}

// STDP Parameters
struct vogels_stdp_plasticity_parameters_struct : stdp_plasticity_parameters_struct {
  vogels_stdp_plasticity_parameters_struct() : tau_istdp(0.02f), learningrate(0.0004f), targetrate(10.0f) { } // default Constructor
  // STDP Parameters
  float tau_istdp;
  float learningrate;
  float targetrate;
  // Alpha must be calculated as 2 * targetrate * tau_istdp
};


class VogelsSTDPPlasticity : public STDPPlasticity {
public:
  VogelsSTDPPlasticity(SpikingSynapses* synapses,
                           SpikingNeurons* neurons,
                           SpikingNeurons* input_neurons,
                           stdp_plasticity_parameters_struct* stdp_parameters);
  ~VogelsSTDPPlasticity() override;
  SPIKE_ADD_BACKEND_GETSET(VogelsSTDPPlasticity, STDPPlasticity);

  struct vogels_stdp_plasticity_parameters_struct* stdp_params;

  void init_backend(Context* ctx = _global_ctx) override;
  void prepare_backend_late() override;

  void state_update(float current_time_in_seconds, float timestep) override;

  // LTP & LTD for this model
  void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep);

private:
  std::shared_ptr<::Backend::VogelsSTDPPlasticity> _backend;
};

#endif
