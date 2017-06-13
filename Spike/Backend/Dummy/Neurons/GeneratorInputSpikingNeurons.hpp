#pragma once

#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class GeneratorInputSpikingNeurons : public virtual ::Backend::Dummy::InputSpikingNeurons,
                                         public virtual ::Backend::GeneratorInputSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(GeneratorInputSpikingNeurons);

      void prepare() override;
      void reset_state() override;

      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::state_update;
    };
  }
}
