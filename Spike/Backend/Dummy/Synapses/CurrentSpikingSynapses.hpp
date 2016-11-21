#pragma once

#include "SpikingSynapses.hpp"
#include "Spike/Synapses/CurrentSpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class CurrentSpikingSynapses : public virtual ::Backend::Dummy::SpikingSynapsesCommon,
                                   public ::Backend::CurrentSpikingSynapses {
    public:
      virtual void prepare() {
        printf("TODO Backend::Dummy::CurrentSpikingSynapses::prepare\n");
      }

      void calculate_postsynaptic_current_injection(::SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {
        printf("TODO Dummy::CurrentSpikingSynapses::calculate_postsynaptic_current_injection\n");
      }

      virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

