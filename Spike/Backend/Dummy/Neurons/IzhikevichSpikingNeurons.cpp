#include "IzhikevichSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, IzhikevichSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void IzhikevichSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
    }
      
    void IzhikevichSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }
  }
}
