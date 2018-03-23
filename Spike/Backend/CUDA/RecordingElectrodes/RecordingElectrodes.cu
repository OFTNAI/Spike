// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/RecordingElectrodes.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, RecordingElectrodes);

namespace Backend {
  namespace CUDA {
    void RecordingElectrodes::prepare() {
      neurons_frontend = frontend()->neurons;
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurons->backend());
      assert(neurons_backend);
    }

    void RecordingElectrodes::reset_state() {
    }
  }
}
