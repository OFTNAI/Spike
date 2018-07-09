#pragma once

#include "Spike/Neurons/ImagePoissonInputSpikingNeurons.hpp"
#include "PoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class ImagePoissonInputSpikingNeurons : public virtual ::Backend::Dummy::PoissonInputSpikingNeurons,
                                            public virtual ::Backend::ImagePoissonInputSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(ImagePoissonInputSpikingNeurons);

      void prepare() override;
      void reset_state() override;

      void copy_rates_to_device() override;
      
      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::state_update;
    };
  }
}
