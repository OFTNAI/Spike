#include "Plasticity.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(Dummy, Plasticity);

namespace Backend {
  namespace Dummy {
    void Plasticity::prepare() {
    }

    void Plasticity::reset_state() {
    }

    void Plasticity::state_update(float current_time_in_seconds, float timestep){}
  }
}
