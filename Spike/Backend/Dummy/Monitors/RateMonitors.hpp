#pragma once

#include "Spike/Monitors/RateMonitors.hpp"
#include "Monitors.hpp"

namespace Backend {
  namespace Dummy {
    class RateMonitors :
      public virtual ::Backend::Dummy::Monitors,
      public virtual ::Backend::RateMonitors {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateMonitors);

      void prepare() override;
      void reset_state() override;

      void add_spikes_to_per_neuron_spike_count
      (float current_time_in_seconds) override;
    };
  }
}
