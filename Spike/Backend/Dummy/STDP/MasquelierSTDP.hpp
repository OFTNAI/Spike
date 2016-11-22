#pragma once

#include "Spike/STDP/MasquelierSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class MasquelierSTDP : public virtual ::Backend::Dummy::STDPCommon,
                           public ::Backend::MasquelierSTDP {
    public:
      virtual void reset_state() {
        // TODO
      }

      virtual void apply_stdp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
        printf("TODO MasquelierSTDP::apply_stdp_to_synapse_weights\n");
      }
    };
  }
}
