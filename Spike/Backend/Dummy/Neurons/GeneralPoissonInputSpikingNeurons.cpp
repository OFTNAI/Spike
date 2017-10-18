#include "GeneralPoissonInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, GeneralPoissonInputSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void GeneralPoissonInputSpikingNeurons::prepare() {
      PoissonInputSpikingNeurons::prepare();
    }
      
    void GeneralPoissonInputSpikingNeurons::reset_state() {
      PoissonInputSpikingNeurons::reset_state();
    }

    void GeneralPoissonInputSpikingNeurons::copy_rates_to_device() {
    }
  }
}
