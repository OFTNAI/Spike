#pragma once

#include "Spike/STDP/HigginsSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class HigginsSTDP : public virtual ::Backend::Dummy::STDP,
                        public virtual ::Backend::HigginsSTDP {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(HigginsSTDP);

      void prepare() override;
      void reset_state() override;

      void apply_ltp_to_synapse_weights(float current_time_in_seconds) override;
      void apply_ltd_to_synapse_weights(float current_time_in_seconds) override;
    };
  }
}
