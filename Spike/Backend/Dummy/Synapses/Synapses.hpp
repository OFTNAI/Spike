#pragma once

#include "Spike/Synapses/Synapses.hpp"

namespace Backend {
  namespace Dummy {
    class Synapses : public virtual ::Backend::Synapses {
    public:
      virtual void set_neuron_indices_by_sampling_from_normal_distribution() {
        printf("TODO Backend::Synapses::set_neuron_indices_by_sampling_from_normal_distribution\n");
      }
      // virtual void prepare() {}
      // virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

