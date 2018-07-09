#include "CustomSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, CustomSTDPPlasticity
);

namespace Backend {
  namespace Dummy {
    void CustomSTDPPlasticity
  ::prepare() {
      STDPPlasticity::prepare();
    }

    void CustomSTDPPlasticity
  ::reset_state() {
      STDPPlasticity::reset_state();
    }

    void CustomSTDPPlasticity
  ::apply_stdp_to_synapse_weights
    (float current_time_in_seconds, float timestep) {
    }
  }
}
