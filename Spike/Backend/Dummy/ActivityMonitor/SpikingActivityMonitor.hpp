#pragma once

#include "Spike/ActivityMonitor/SpikingActivityMonitor.hpp"
#include "ActivityMonitor.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingActivityMonitor :
      public virtual ::Backend::Dummy::ActivityMonitor,
      public virtual ::Backend::SpikingActivityMonitor {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikingActivityMonitor);

      void prepare() override;
      void reset_state() override;

      void copy_spikes_to_front() override;
      void copy_spikecount_to_front() override;
      void collect_spikes_for_timestep
      (float current_time_in_seconds, float timestep) override;
    };
  }
}
