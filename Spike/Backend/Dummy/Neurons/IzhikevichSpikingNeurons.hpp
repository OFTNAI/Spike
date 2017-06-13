#pragma once

#include "Spike/Neurons/IzhikevichSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class IzhikevichSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
                                     public virtual ::Backend::IzhikevichSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(IzhikevichSpikingNeurons);

      void prepare() override;
      void reset_state() override;

      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::state_update;
    };
  }
}
