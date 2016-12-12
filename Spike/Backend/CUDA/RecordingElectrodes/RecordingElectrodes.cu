// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/RecordingElectrodes.hpp"
#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    void RecordingElectrodes::prepare() {
      neurons_frontend = frontend()->neurons;
      // TODO: Fix this weird cludge (want to cast most generically!)
      neurons_backend = dynamic_cast<::Backend::CUDA::LIFSpikingNeurons*>
        (frontend()->neurons->backend());
    }
  }
}
