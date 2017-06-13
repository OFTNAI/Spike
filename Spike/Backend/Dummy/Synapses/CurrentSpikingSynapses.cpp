#include "CurrentSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, CurrentSpikingSynapses);

namespace Backend {
  namespace Dummy {
    void CurrentSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
    }

    void CurrentSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
    }
  }
}
