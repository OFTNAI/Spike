// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Monitors/Monitors.hpp"

//SPIKE_EXPORT_BACKEND_TYPE(CUDA, Monitors);

namespace Backend {
  namespace CUDA {
    void Monitors::prepare() {
      neurons_frontend = frontend()->neurons;
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurons->backend());
      assert(neurons_backend);
    }

    void Monitors::reset_state() {
    }
  }
}
