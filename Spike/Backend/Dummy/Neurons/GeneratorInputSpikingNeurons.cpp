#include "GeneratorInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, GeneratorInputSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void GeneratorInputSpikingNeurons::prepare() {
      InputSpikingNeurons::prepare();
    }
      
    void GeneratorInputSpikingNeurons::reset_state() {
      InputSpikingNeurons::reset_state();
    }
  }
}
