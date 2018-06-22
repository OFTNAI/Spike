#include "RateMonitors.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, RateMonitors);

namespace Backend {
  namespace Dummy {
    void RateMonitors::prepare() {
      Monitors::prepare();
    }

    void RateMonitors::reset_state() {
      Monitors::reset_state();
    }

    void RateMonitors::add_spikes_to_per_neuron_spike_count
    (float current_time_in_seconds) {
    }
  }
}
