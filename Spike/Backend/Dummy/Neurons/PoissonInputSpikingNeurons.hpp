#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class PoissonInputSpikingNeurons : public virtual ::Backend::Dummy::InputSpikingNeurons,
                                       public virtual ::Backend::PoissonInputSpikingNeurons {
    public:
      PoissonInputSpikingNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(PoissonInputSpikingNeurons);

      void prepare() override;
      void reset_state() override;
    };
  }
}
