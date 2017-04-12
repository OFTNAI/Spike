#pragma once

#include "Spike/STDP/vanRossumSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class vanRossumSTDP : public virtual ::Backend::Dummy::STDP,
                           public virtual ::Backend::vanRossumSTDP {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(vanRossumSTDP);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(float current_time_in_seconds) override;
    };
  }
}
