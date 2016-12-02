#pragma once

#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class CurrentSpikingSynapses : public virtual SpikingSynapsesCommon,
                                   public ::Backend::CurrentSpikingSynapses {
    public:
      MAKE_BACKEND_CONSTRUCTOR(CurrentSpikingSynapses);

      virtual void prepare() {
        printf("TODO Backend::Dummy::CurrentSpikingSynapses::prepare\n");
      }

      void calculate_postsynaptic_current_injection(::SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {
        printf("TODO Dummy::CurrentSpikingSynapses::calculate_postsynaptic_current_injection\n");
      }

      virtual void reset_state() {}

      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  } // namespace Dummy
} // namespace Backend

