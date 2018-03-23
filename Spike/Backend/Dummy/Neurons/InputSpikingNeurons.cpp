#include "InputSpikingNeurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(Dummy, InputSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void InputSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
    }
      
    void InputSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }
  }
}

