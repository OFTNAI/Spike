#pragma once

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingNeurons : public virtual ::Backend::Dummy::Neurons,
                           public virtual ::Backend::SpikingNeurons {
    public:
      void prepare() override;
      void reset_state() override;

      void state_update(float current_time_in_seconds,
                                   float timestep) override;
    };
  } // namespace Dummy
} // namespace Backend
