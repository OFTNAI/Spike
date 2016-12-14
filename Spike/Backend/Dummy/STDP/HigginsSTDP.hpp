#pragma once

#include "Spike/STDP/HigginsSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class HigginsSTDP : public virtual ::Backend::Dummy::STDP,
                        public virtual ::Backend::HigginsSTDP {
    public:
      MAKE_BACKEND_CONSTRUCTOR(HigginsSTDP);

      void prepare() override {
        STDP::prepare();
      }

      void reset_state() override {
        STDP::reset_state();
      }

      void push_data_front() override {
        STDP::push_data_front();
      }

      void pull_data_back() override {
        STDP::pull_data_back();
      }

      void apply_ltp_to_synapse_weights(float current_time_in_seconds) override {
      }

      void apply_ltd_to_synapse_weights(float current_time_in_seconds) override {
      }
    };
  }
}
