#pragma once

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingNeurons : public virtual ::Backend::Dummy::NeuronsCommon,
                           public ::Backend::SpikingNeurons {
    public:
      virtual void reset_state();
      virtual void copy_constants_to_device();
    };
  } // namespace Dummy
} // namespace Backend
