#pragma once

#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class CurrentSpikingSynapses : public virtual ::Backend::Dummy::SpikingSynapses,
                                   public virtual ::Backend::CurrentSpikingSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CurrentSpikingSynapses);

      void prepare() override;
      void reset_state() override;

      void calculate_postsynaptic_current_injection
      (::SpikingNeurons * neurons,
       float current_time_in_seconds, float timestep) final;
    };
  } // namespace Dummy
} // namespace Backend

