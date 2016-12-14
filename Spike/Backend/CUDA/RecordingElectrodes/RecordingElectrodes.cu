// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/RecordingElectrodes.hpp"

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

    void RecordingElectrodes::push_data_front() {
    }

    void RecordingElectrodes::pull_data_back() {
    }
  }
}
