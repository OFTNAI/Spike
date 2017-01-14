#include "CurrentSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, CurrentSpikingSynapses);

namespace Backend {
  namespace Dummy {
    void CurrentSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
    }

    void CurrentSpikingSynapses::calculate_postsynaptic_current_injection
    (::SpikingNeurons * neurons,
     float current_time_in_seconds, float timestep) {
    }

    void CurrentSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
    }
  }
}
