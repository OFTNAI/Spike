#pragma once

#include "Spike/Neurons/InputSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    using InputSpikingNeurons = ::Backend::Dummy::SpikingNeurons;
    // class InputSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
    //                             public virtual ::Backend::InputSpikingNeurons {
    // public:
    // };
  }
}
