#pragma once

#include "Spike/STDP/MasquelierSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class MasquelierSTDP : public virtual ::Backend::Dummy::STDP,
                           public virtual ::Backend::MasquelierSTDP {
    public:
      MAKE_BACKEND_CONSTRUCTOR(MasquelierSTDP);

      virtual void reset_state() {
        // TODO
      }

      virtual void apply_stdp_to_synapse_weights(float current_time_in_seconds) {
        printf("TODO MasquelierSTDP::apply_stdp_to_synapse_weights\n");
      }

      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  }
}
