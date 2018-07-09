#pragma once

#include "Spike/Plasticity/WeightDependentSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class WeightDependentSTDPPlasticity : public virtual ::Backend::Dummy::STDPPlasticity,
                           public virtual ::Backend::WeightDependentSTDPPlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(WeightDependentSTDPPlasticity);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
    };
  }
}
