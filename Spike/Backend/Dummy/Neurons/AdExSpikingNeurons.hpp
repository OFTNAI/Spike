#pragma once

#include "Spike/Neurons/AdExSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class AdExSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
                               public virtual ::Backend::AdExSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AdExSpikingNeurons);

      void prepare() override;
      void reset_state() override;

      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::state_update;
    };
  }
}
