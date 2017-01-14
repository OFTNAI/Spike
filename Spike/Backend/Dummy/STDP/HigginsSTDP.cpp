#include "HigginsSTDP.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, HigginsSTDP);

namespace Backend {
  namespace Dummy {
    void HigginsSTDP::prepare() {
      STDP::prepare();
    }

    void HigginsSTDP::reset_state() {
      STDP::reset_state();
    }

    void HigginsSTDP::apply_ltp_to_synapse_weights
    (float current_time_in_seconds) {
    }

    void HigginsSTDP::apply_ltd_to_synapse_weights
    (float current_time_in_seconds) {
    }
  }
}
