#pragma once

#include "Spike/STDP/EvansSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class EvansSTDP : public virtual ::Backend::Dummy::STDP,
                      public virtual ::Backend::EvansSTDP {
    public:
      MAKE_BACKEND_CONSTRUCTOR(EvansSTDP);

      void prepare() override {
        // TODO
      }

      void reset_state() override {
        // TODO
      }

      void update_synaptic_efficacies_or_weights(float current_time_in_seconds) override {
        printf("TODO EvansSTDP::update_synaptic_efficacies_or_weights\n");
      }

      void update_presynaptic_activities(float timestep, float current_time_in_seconds) override {
        printf("TODO EvansSTDP::update_presynaptic_activities\n");
      }

      void update_postsynaptic_activities(float timestep, float current_time_in_seconds) override {
        printf("TODO EvansSTDP::update_postsynaptic_activities\n");
      }

      void push_data_front() override {}
      void pull_data_back() override {}
    };
  }
}
