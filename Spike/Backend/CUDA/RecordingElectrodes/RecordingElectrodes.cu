// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/RecordingElectrodes.hpp"

namespace Backend {
  namespace CUDA {
    void RecordingElectrodes::prepare() {
      neurons_frontend = frontend()->neurons;
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurons->backend());
    }
  }
}
