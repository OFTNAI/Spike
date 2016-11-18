#pragma once

#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class GeneratorInputSpikingNeurons : public ::Backend::GeneratorInputSpikingNeurons {
    public:
      virtual void reset_state();
    };
  }
}
