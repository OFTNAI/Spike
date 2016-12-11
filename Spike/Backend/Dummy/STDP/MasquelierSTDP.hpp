#pragma once

#include "Spike/STDP/MasquelierSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class MasquelierSTDP : public virtual ::Backend::Dummy::STDP,
                           public virtual ::Backend::MasquelierSTDP {
    public:
      MAKE_BACKEND_CONSTRUCTOR(MasquelierSTDP);

      void prepare() override {
        // TODO
      }

      void reset_state() override {
        // TODO
      }

      void apply_stdp_to_synapse_weights(float current_time_in_seconds) override {
        printf("TODO MasquelierSTDP::apply_stdp_to_synapse_weights\n");
      }

      void push_data_front() override {}
      void pull_data_back() override {}
    };
  }
}
