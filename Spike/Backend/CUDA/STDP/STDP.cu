// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/STDP.hpp"
#include <iostream>

namespace Backend {
  namespace CUDA {
    void STDP::prepare() {
      std::cout << "????4 " << frontend() << "  "
                << frontend()->neurs << "\n";    
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurs->backend());
      synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>
        (frontend()->syns->backend());
    }
  }
}