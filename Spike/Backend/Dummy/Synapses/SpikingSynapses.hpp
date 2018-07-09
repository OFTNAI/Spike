#pragma once

#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Synapses.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingSynapses : public virtual ::Backend::Dummy::Synapses,
                            public virtual ::Backend::SpikingSynapses {
    public:
      SpikingSynapses();
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikingSynapses);
      void prepare() override;
      void reset_state() override;

      void copy_weights_to_host() override;
      void state_update
      (::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons,
       float current_time_in_seconds, float timestep) final;
    };
  } // namespace Dummy
} // namespace Backend

