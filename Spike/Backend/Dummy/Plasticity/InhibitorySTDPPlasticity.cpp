#include "InhibitorySTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, InhibitorySTDPPlasticity
);

namespace Backend {
  namespace Dummy {
    void InhibitorySTDPPlasticity
  ::prepare() {
      STDPPlasticity::prepare();
    }

    void InhibitorySTDPPlasticity
  ::reset_state() {
      STDPPlasticity::reset_state();
    }

    void InhibitorySTDPPlasticity
  ::apply_stdp_to_synapse_weights
    (float current_time_in_seconds, float timestep) {
    }
  }
}
