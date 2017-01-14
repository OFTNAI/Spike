#pragma once

#include "Spike/STDP/MasquelierSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class MasquelierSTDP : public virtual ::Backend::Dummy::STDP,
                           public virtual ::Backend::MasquelierSTDP {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(MasquelierSTDP);

      void prepare() override {
        STDP::prepare();
      }

      void reset_state() override {
        STDP::reset_state();
      }

      void apply_stdp_to_synapse_weights(float current_time_in_seconds) override {
      }
    };
  }
}
