// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/STDP.hpp"

namespace Backend {
  namespace CUDA {
    void STDP::prepare() {
      assert("This is a test; please remove this line" && false);
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurs->backend());
      synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>
        (frontend()->syns->backend());
    }
  }
}
