#include "vanRossumSTDP.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, vanRossumSTDP
);

namespace Backend {
  namespace Dummy {
    void vanRossumSTDP
  ::prepare() {
      STDP::prepare();
    }

    void vanRossumSTDP
  ::reset_state() {
      STDP::reset_state();
    }

    void vanRossumSTDP
  ::apply_stdp_to_synapse_weights
    (float current_time_in_seconds) {
    }
  }
}
