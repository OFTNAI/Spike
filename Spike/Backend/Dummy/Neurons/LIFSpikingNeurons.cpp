#include "LIFSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, LIFSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void LIFSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
    }
      
    void LIFSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }
  }
}
