// Weight Normalization (Spiking) Class Header
// SpikingWeightNormPlasiticity.hpp
//
//  Author: Nasir Ahmad
//  Date: 29/05/2016

#ifndef SPIKINGWEIGHTNORMPLASTICITY_H
#define SPIKINGWEIGHTNORMPLASTICITY_H

class WeightNormSTDPPlasticity; // forward definition

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Plasticity/STDPPlasticity.hpp"
#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

namespace Backend {
  class WeightNormSTDPPlasticity : public virtual STDPPlasticity {
  public:
    SPIKE_ADD_BACKEND_FACTORY(WeightNormSTDPPlasticity);

    virtual void weight_normalization() = 0;
  };
}

static_assert(std::has_virtual_destructor<Backend::WeightNormSTDPPlasticity>::value,
              "contract violated");

// WeightNormSTDPPlasticity Parameters
struct weightnorm_stdp_plasticity_parameters_struct : stdp_plasticity_parameters_struct {
  weightnorm_stdp_plasticity_parameters_struct() {}
  // The normalization can be either done with the initialized total or with a specific target
  bool settarget = false;
  float target = 0.0;
};


class WeightNormSTDPPlasticity : public STDPPlasticity {
public:
  WeightNormSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, plasticity_parameters_struct* parameters);
  ~WeightNormSTDPPlasticity() override;

  SPIKE_ADD_BACKEND_GETSET(WeightNormSTDPPlasticity, SpikeBase);
  void reset_state() override;

  weightnorm_stdp_plasticity_parameters_struct* plasticity_parameters = nullptr;

  float* sum_squared_afferent_values = nullptr;
  float* afferent_weight_change_updater = nullptr;
  float* initial_weights = nullptr;
  bool* neuron_in_plasticity_set = nullptr;

  void init_backend(Context* ctx = _global_ctx) override;
  void prepare_backend_early() override;
  virtual void state_update(float current_time_in_seconds, float timestep) override;

private:
  std::shared_ptr<::Backend::WeightNormSTDPPlasticity> _backend;
};

#endif
