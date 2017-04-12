#include "VogelsSTDP.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, VogelsSTDP
);

namespace Backend {
  namespace Dummy {
    void VogelsSTDP
  ::prepare() {
      STDP::prepare();
    }

    void VogelsSTDP
  ::reset_state() {
      STDP::reset_state();
    }

    void VogelsSTDP
  ::apply_stdp_to_synapse_weights
    (float current_time_in_seconds, float timestep) {
    }
  }
}
