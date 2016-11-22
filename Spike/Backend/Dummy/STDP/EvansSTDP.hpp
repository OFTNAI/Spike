#pragma once

#include "Spike/STDP/EvansSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class EvansSTDP : public virtual ::Backend::Dummy::STDPCommon,
                      public ::Backend::EvansSTDP {
    public:
      virtual void reset_state() {
        // TODO
      }

      virtual void update_synaptic_efficacies_or_weights(float current_time_in_seconds, float * d_last_spike_time_of_each_neuron) {
        printf("TODO EvansSTDP::update_synaptic_efficacies_or_weights\n");
      }

      virtual void update_presynaptic_activities(float timestep, float current_time_in_seconds) {
        printf("TODO EvansSTDP::update_presynaptic_activities\n");
      }

      virtual void update_postsynaptic_activities(float timestep, float current_time_in_seconds) {
        printf("TODO EvansSTDP::update_postsynaptic_activities\n");
      }
    };
  }
}
