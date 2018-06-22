#pragma once

#include "Spike/Monitors/SpikeMonitors.hpp"
#include "Monitors.hpp"

namespace Backend {
  namespace Dummy {
    class SpikeMonitors :
      public virtual ::Backend::Dummy::Monitors,
      public virtual ::Backend::SpikeMonitors {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikeMonitors);

      void prepare() override;
      void reset_state() override;

      void copy_spikes_to_front() override;
      void copy_spikecount_to_front() override;
      void collect_spikes_for_timestep
      (float current_time_in_seconds, float timestep) override;
    };
  }
}
