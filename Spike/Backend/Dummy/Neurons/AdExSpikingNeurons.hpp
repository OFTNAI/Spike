#pragma once

#include "Spike/Neurons/AdExSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class AdExSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
                               public virtual ::Backend::AdExSpikingNeurons {
    public:
      MAKE_BACKEND_CONSTRUCTOR(AdExSpikingNeurons);

      virtual void reset_state();
      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  }
}
