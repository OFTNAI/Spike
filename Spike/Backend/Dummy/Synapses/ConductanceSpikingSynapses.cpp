#include "ConductanceSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, ConductanceSpikingSynapses);

namespace Backend {
  namespace Dummy {
    void ConductanceSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
    }

    void ConductanceSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
    }

    void ConductanceSpikingSynapses::calculate_postsynaptic_current_injection
    (::SpikingNeurons * neurons,
     float current_time_in_seconds,
     float timestep) {
    }

    void ConductanceSpikingSynapses::update_synaptic_conductances
    (float timestep, float current_time_in_seconds) {
    }
  }
}
