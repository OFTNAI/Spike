#include "RateActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, RateActivityMonitor);

namespace Backend {
  namespace Dummy {
    void RateActivityMonitor::prepare() {
      ActivityMonitor::prepare();
    }

    void RateActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
    }

    void RateActivityMonitor::add_spikes_to_per_neuron_spike_count
    (float current_time_in_seconds, float timestep) {
    }
      
    void RateActivityMonitor::copy_spike_count_to_host() {
    }
  }
}
