#include "SpikingSynapses.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(Dummy, SpikingSynapses);

namespace Backend {
  namespace Dummy {
    void SpikingSynapses::prepare() {
      Synapses::prepare();
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();
    }

    void SpikingSynapses::state_update
    (::SpikingNeurons * neurons,
     ::SpikingNeurons * input_neurons,
     float current_time_in_seconds, float timestep) {
    }

    void SpikingSynapses::copy_weights_to_host() {
    }
  }
}
