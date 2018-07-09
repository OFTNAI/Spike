#include "VoltageSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, VoltageSpikingSynapses);

namespace Backend {
  namespace Dummy {
    void VoltageSpikingSynapses::prepare() {
      SpikingSynapses::prepare();
    }

    void VoltageSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
    }

  }
}
