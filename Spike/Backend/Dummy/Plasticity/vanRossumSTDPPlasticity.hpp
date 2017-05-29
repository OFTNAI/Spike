#pragma once

#include "Spike/Plasticity/vanRossumSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class vanRossumSTDPPlasticity : public virtual ::Backend::Dummy::STDPPlasticity,
                           public virtual ::Backend::vanRossumSTDPPlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(vanRossumSTDPPlasticity);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(float current_time_in_seconds) override;
    };
  }
}
