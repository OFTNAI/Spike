#pragma once

#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Synapses.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingSynapses : public virtual ::Backend::Dummy::Synapses,
                            public virtual ::Backend::SpikingSynapses {
    public:
      void interact_spikes_with_synapses(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) final {
      }

      void prepare() override {
        Synapses::prepare();
      }

      void reset_state() override {
        Synapses::reset_state();
      }

      void push_data_front() override {
        Synapses::push_data_front();
      }

      void pull_data_back() override {
        Synapses::pull_data_back();
      }
    };
  } // namespace Dummy
} // namespace Backend

