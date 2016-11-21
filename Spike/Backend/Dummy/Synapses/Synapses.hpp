#pragma once

#include "Spike/Synapses/Synapses.hpp"

namespace Backend {
  namespace Dummy {
    class SynapsesCommon : public virtual ::Backend::SynapsesCommon {
    public:
      virtual void set_neuron_indices_by_sampling_from_normal_distribution() {
        printf("TODO Backend::Synapses::set_neuron_indices_by_sampling_from_normal_distribution\n");
      }
    };

    class Synapses : public ::Backend::Synapses,
                     public virtual SynapsesCommon {
    public:
      // virtual void prepare() {}
      // virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

