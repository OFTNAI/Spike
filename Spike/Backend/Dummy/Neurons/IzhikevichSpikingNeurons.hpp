#pragma once

#include "Spike/Neurons/IzhikevichSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class IzhikevichSpikingNeurons : public ::Backend::IzhikevichSpikingNeurons {
    public:

      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
      virtual void reset_state();

    };
  }
}
