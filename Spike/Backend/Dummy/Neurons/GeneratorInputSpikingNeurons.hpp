#pragma once

#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class GeneratorInputSpikingNeurons : public virtual ::Backend::Dummy::NeuronsCommon,
                                         public ::Backend::GeneratorInputSpikingNeurons {
    public:
      virtual void reset_state();
    };
  }
}
