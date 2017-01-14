#include "Neurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(Dummy, Neurons);

namespace Backend {
  namespace Dummy {
    void Neurons::prepare() {
    }

    void Neurons::reset_state() {
      reset_current_injections();
    }

    void Neurons::reset_current_injections() {
    }
  } // namespace Dummy
} // namespace Backend
