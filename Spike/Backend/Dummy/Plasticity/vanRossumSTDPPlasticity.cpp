#include "vanRossumSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, vanRossumSTDPPlasticity
);

namespace Backend {
  namespace Dummy {
    void vanRossumSTDPPlasticity
  ::prepare() {
      STDPPlasticity::prepare();
    }

    void vanRossumSTDPPlasticity
  ::reset_state() {
      STDPPlasticity::reset_state();
    }

    void vanRossumSTDPPlasticity
  ::apply_stdp_to_synapse_weights
    (float current_time_in_seconds) {
    }
  }
}
