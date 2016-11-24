#pragma once

#include "Spike/Neurons/AdExSpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class AdExSpikingNeurons : public virtual ::Backend::Dummy::NeuronsCommon,
                               public ::Backend::AdExSpikingNeurons {
    public:
      virtual void reset_state();
    };
  }
}
