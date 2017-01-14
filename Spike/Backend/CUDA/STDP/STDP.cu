// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/STDP.hpp"
#include <iostream>

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, STDP);

namespace Backend {
  namespace CUDA {
    void STDP::prepare() {
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurs->backend());
      synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>
        (frontend()->syns->backend());
    }

    void STDP::reset_state() {
    }
  }
}
