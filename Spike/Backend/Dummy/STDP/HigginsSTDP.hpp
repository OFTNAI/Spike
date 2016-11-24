#pragma once

#include "Spike/STDP/HigginsSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class HigginsSTDP : public virtual ::Backend::Dummy::STDPCommon,
                        public ::Backend::HigginsSTDP {
    public:
      MAKE_BACKEND_CONSTRUCTOR(HigginsSTDP);

      virtual void reset_state() {
        // TODO
      }

      virtual void apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
        printf("TODO HigginsSTDP::apply_ltp_to_synapse_weights\n");
      }

      virtual void apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
        printf("TODO HigginsSTDP::apply_ltd_to_synapse_weights\n");
      }
    };
  }
}
