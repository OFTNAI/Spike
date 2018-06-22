#pragma once

#include "Spike/Neurons/PatternedPoissonInputSpikingNeurons.hpp"
#include "PoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class PatternedPoissonInputSpikingNeurons : public virtual ::Backend::Dummy::PoissonInputSpikingNeurons,
                                            public virtual ::Backend::PatternedPoissonInputSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(PatternedPoissonInputSpikingNeurons);

      void prepare() override;
      void reset_state() override;

      void copy_rates_to_device() override;
      
      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::state_update;
    };
  }
}
