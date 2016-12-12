// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/InputSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    void InputSpikingNeurons::reset_state() {
      printf("!!! TODO InputSpikingNeurons::reset_state called\n");
    }

    void InputSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
    }
  }
}
