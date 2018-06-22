#pragma once

#include "Spike/ActivityMonitor/RateActivityMonitor.hpp"
#include "ActivityMonitor.hpp"

namespace Backend {
  namespace Dummy {
    class RateActivityMonitor :
      public virtual ::Backend::Dummy::ActivityMonitor,
      public virtual ::Backend::RateActivityMonitor {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateActivityMonitor);

      void prepare() override;
      void reset_state() override;

      void add_spikes_to_per_neuron_spike_count
      (float current_time_in_seconds) override;
    };
  }
}
