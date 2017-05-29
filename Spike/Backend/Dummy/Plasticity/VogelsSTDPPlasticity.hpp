#pragma once

#include "Spike/Plasticity/VogelsSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class VogelsSTDPPlasticity : public virtual ::Backend::Dummy::STDPPlasticity,
                           public virtual ::Backend::VogelsSTDPPlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(VogelsSTDPPlasticity);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
    };
  }
}
