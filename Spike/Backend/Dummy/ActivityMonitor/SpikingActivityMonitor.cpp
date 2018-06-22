#include "SpikingMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, SpikingMonitor);

namespace Backend {
  namespace Dummy {
    void SpikingMonitor::prepare() {
      AcitivtyMonitor::prepare();
    }

    void SpikingMonitor::reset_state() {
      AcitivtyMonitor::reset_state();
    }
    
    void SpikingMonitor::copy_spikecount_to_front() {
    }

    void SpikingMonitor::copy_spikes_to_front() {
    }

    void SpikingMonitor::collect_spikes_for_timestep
    (float current_time_in_seconds, float timestep) {
    }
  }
}
