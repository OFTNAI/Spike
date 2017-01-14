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

    void SpikingNeurons::check_for_neuron_spikes
    (float current_time_in_seconds, float timestep) {
    }

    void SpikingNeurons::update_membrane_potentials
    (float timestep, float current_time_in_seconds) {
    }
  } // namespace Dummy
} // namespace Backend

