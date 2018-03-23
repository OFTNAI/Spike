#include "SpikingNeurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(Dummy, SpikingNeurons);

namespace Backend {
  namespace Dummy {
    void SpikingNeurons::prepare() {
      Neurons::prepare();
    }

    void SpikingNeurons::reset_state() {
      Neurons::reset_state();
    }

    void SpikingNeurons::state_update
    (float current_time_in_seconds, float timestep) {
    }

  } // namespace Dummy
} // namespace Backend

