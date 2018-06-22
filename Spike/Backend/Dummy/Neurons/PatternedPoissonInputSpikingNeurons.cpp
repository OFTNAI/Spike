#include "PatternedPoissonInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, PatternedPoissonInputSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void PatternedPoissonInputSpikingNeurons::prepare() {
      PoissonInputSpikingNeurons::prepare();
    }
      
    void PatternedPoissonInputSpikingNeurons::reset_state() {
      PoissonInputSpikingNeurons::reset_state();
    }

    void PatternedPoissonInputSpikingNeurons::copy_rates_to_device() {
    }
  }
}
