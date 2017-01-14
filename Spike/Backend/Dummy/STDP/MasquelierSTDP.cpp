#include "MasquelierSTDP.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, MasquelierSTDP);

namespace Backend {
  namespace Dummy {
    void MasquelierSTDP::prepare() {
      STDP::prepare();
    }

    void MasquelierSTDP::reset_state() {
      STDP::reset_state();
    }

    void MasquelierSTDP::apply_stdp_to_synapse_weights
    (float current_time_in_seconds) {
    }
  }
}
