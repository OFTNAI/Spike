// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/STDP.hpp"
#include <iostream>

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

    void STDP::push_data_front() {
    }

    void STDP::pull_data_back() {
    }
  }
}
