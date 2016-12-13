#pragma once

#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class ConductanceSpikingSynapses : public virtual ::Backend::Dummy::SpikingSynapses,
                                       public virtual ::Backend::ConductanceSpikingSynapses {
    public:
      MAKE_BACKEND_CONSTRUCTOR(ConductanceSpikingSynapses);
      
      void prepare() override {
        SpikingSynapses::prepare();
      }

      void calculate_postsynaptic_current_injection
      (::SpikingNeurons * neurons,
       float current_time_in_seconds,
       float timestep) final {
      }

      void update_synaptic_conductances(float timestep, float current_time_in_seconds) final {
      }

      void reset_state() override {
        SpikingSynapses::reset_state();
      }

      void push_data_front() override {
        SpikingSynapses::push_data_front();
      }

      void pull_data_back() override {
        SpikingSynapses::pull_data_back();
      }
    };
  } // namespace Dummy
} // namespace Backend

