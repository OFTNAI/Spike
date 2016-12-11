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
        // TODO
      }

      void reset_state() override {
        // TODO
      }

      void apply_ltp_to_synapse_weights(float current_time_in_seconds) override {
        printf("TODO HigginsSTDP::apply_ltp_to_synapse_weights\n");
      }

      void apply_ltd_to_synapse_weights(float current_time_in_seconds) override {
        printf("TODO HigginsSTDP::apply_ltd_to_synapse_weights\n");
      }

      void push_data_front() override {}
      void pull_data_back() override {}
    };
  }
}
