#pragma once

#include "Spike/Plasticity/CustomSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class CustomSTDPPlasticity : public virtual ::Backend::Dummy::STDPPlasticity,
                           public virtual ::Backend::CustomSTDPPlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CustomSTDPPlasticity);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
    };
  }
}
