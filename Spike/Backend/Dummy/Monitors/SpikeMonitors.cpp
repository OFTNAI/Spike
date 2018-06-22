#include "SpikeMonitors.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, SpikeMonitors);

namespace Backend {
  namespace Dummy {
    void SpikeMonitors::prepare() {
      Monitors::prepare();
    }

    void SpikeMonitors::reset_state() {
      Monitors::reset_state();
    }
    
    void SpikeMonitors::copy_spikecount_to_front() {
    }

    void SpikeMonitors::copy_spikes_to_front() {
    }

    void SpikeMonitors::collect_spikes_for_timestep
    (float current_time_in_seconds, float timestep) {
    }
  }
}
