#pragma once

#include "Spike/STDP/EvansSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class EvansSTDP : public virtual ::Backend::Dummy::STDP,
                      public virtual ::Backend::EvansSTDP {
    public:
      MAKE_BACKEND_CONSTRUCTOR(EvansSTDP);

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

      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  }
}
