#pragma once

#include "Spike/Neurons/InputSpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class InputSpikingNeurons : public virtual ::Backend::Dummy::NeuronsCommon,
                                public ::Backend::InputSpikingNeurons {
    public:
    };
  }
}
