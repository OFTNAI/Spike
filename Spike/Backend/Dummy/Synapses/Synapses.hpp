#pragma once

#include "Spike/Synapses/Synapses.hpp"

namespace Backend {
  namespace Dummy {
    class Synapses : public virtual ::Backend::Synapses {
    public:
      ~Synapses() override = default;

      void prepare() override;
      void reset_state() override;

      void set_neuron_indices_by_sampling_from_normal_distribution
      (int original_number_of_synapses,
       int total_number_of_new_synapses,
       int postsynaptic_group_id,
       int poststart, int prestart,
       int* postsynaptic_group_shape,
       int* presynaptic_group_shape,
       int number_of_new_synapses_per_postsynaptic_neuron,
       int number_of_postsynaptic_neurons_in_group,
       int max_number_of_connections_per_pair,
       float standard_deviation_sigma,
       bool presynaptic_group_is_input) final;
    };
  } // namespace Dummy
} // namespace Backend

