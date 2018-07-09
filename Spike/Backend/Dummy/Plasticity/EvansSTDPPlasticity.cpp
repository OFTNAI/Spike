#include "EvansSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, EvansSTDPPlasticity);

namespace Backend {
  namespace Dummy {
    void EvansSTDPPlasticity::prepare() {
      STDPPlasticity::prepare();
    }

    void EvansSTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();
    }

    void EvansSTDPPlasticity::update_synaptic_efficacies_or_weights
    (float current_time_in_seconds) {
    }

    void EvansSTDPPlasticity::update_presynaptic_activities
    (float timestep, float current_time_in_seconds) {
    }

    void EvansSTDPPlasticity::update_postsynaptic_activities
    (float timestep, float current_time_in_seconds) {
    }
  }
}
