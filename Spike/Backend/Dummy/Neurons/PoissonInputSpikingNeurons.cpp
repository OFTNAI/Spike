#include "PoissonInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, PoissonInputSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void PoissonInputSpikingNeurons::prepare() {
      InputSpikingNeurons::prepare();
    }
      
    void PoissonInputSpikingNeurons::reset_state() {
      InputSpikingNeurons::reset_state();
    }
  }
}
