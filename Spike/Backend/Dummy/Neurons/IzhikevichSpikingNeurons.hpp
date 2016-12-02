#pragma once

#include "Spike/Neurons/IzhikevichSpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class IzhikevichSpikingNeurons : public virtual ::Backend::Dummy::NeuronsCommon,
                                     public ::Backend::IzhikevichSpikingNeurons {
    public:
      MAKE_BACKEND_CONSTRUCTOR(IzhikevichSpikingNeurons);

      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
      virtual void reset_state();
      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  }
}
