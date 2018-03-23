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

  }
}
