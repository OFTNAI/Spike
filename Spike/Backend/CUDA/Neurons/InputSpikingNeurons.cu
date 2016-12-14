// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/InputSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    void InputSpikingNeurons::reset_state() {
      SpikingNeurons::reset_state();
    }

    void InputSpikingNeurons::prepare() {
      SpikingNeurons::prepare();
    }

    void InputSpikingNeurons::push_data_front() {
      SpikingNeurons::push_data_front();
    }

    void InputSpikingNeurons::pull_data_back() {
      SpikingNeurons::pull_data_back();
    }
  }
}
