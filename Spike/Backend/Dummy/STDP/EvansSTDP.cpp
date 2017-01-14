#include "EvansSTDP.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, EvansSTDP);

namespace Backend {
  namespace Dummy {
    void EvansSTDP::prepare() {
      STDP::prepare();
    }

    void EvansSTDP::reset_state() {
      STDP::reset_state();
    }

    void EvansSTDP::update_synaptic_efficacies_or_weights
    (float current_time_in_seconds) {
    }

    void EvansSTDP::update_presynaptic_activities
    (float timestep, float current_time_in_seconds) {
    }

    void EvansSTDP::update_postsynaptic_activities
    (float timestep, float current_time_in_seconds) {
    }
  }
}
