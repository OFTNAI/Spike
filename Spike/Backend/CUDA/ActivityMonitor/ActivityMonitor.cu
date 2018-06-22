// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/ActivityMonitor/ActivityMonitor.hpp"

//SPIKE_EXPORT_BACKEND_TYPE(CUDA, ActivityMonitor);

namespace Backend {
  namespace CUDA {
    void ActivityMonitor::prepare() {
      neurons_frontend = frontend()->neurons;
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurons->backend());
      assert(neurons_backend);
    }

    void ActivityMonitor::reset_state() {
    }
  }
}
