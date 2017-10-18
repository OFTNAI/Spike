#pragma once

#include "Spike/Neurons/GeneralPoissonInputSpikingNeurons.hpp"
#include "PoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class GeneralPoissonInputSpikingNeurons : public virtual ::Backend::Dummy::PoissonInputSpikingNeurons,
                                            public virtual ::Backend::GeneralPoissonInputSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(GeneralPoissonInputSpikingNeurons);

      void prepare() override;
      void reset_state() override;

      void copy_rates_to_device() override;
      
      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::state_update;
    };
  }
}
