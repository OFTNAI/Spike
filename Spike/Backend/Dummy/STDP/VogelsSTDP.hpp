#pragma once

#include "Spike/STDP/VogelsSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class VogelsSTDP : public virtual ::Backend::Dummy::STDP,
                           public virtual ::Backend::VogelsSTDP {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(VogelsSTDP);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
    };
  }
}
