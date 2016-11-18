#pragma once

#include "Spike/Neurons/AdExSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class AdExSpikingNeurons : public ::Backend::AdExSpikingNeurons {
    public:
      virtual void reset_state();
    };
  }
}
