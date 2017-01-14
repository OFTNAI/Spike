// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/InputSpikingNeurons.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, InputSpikingNeurons);

namespace Backend {
  namespace CUDA {
    void InputSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }

    void InputSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
    }
  }
}
