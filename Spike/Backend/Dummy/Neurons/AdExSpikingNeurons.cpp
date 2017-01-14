#include "AdExSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, AdExSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void AdExSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
    }
      
    void AdExSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }
  }
}

