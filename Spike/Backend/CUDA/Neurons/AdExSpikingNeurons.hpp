#pragma once

#include "Spike/Neurons/AdExSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include <cuda.h>
#include <vector_types.h>

namespace Backend {
  namespace CUDA {
    class AdExSpikingNeurons : public virtual ::Backend::CUDA::SpikingNeurons,
                               public virtual ::Backend::AdExSpikingNeurons {
    public:
      float * adaptation_values_w = nullptr;
      float * membrane_capacitances_Cm = nullptr;
      float * membrane_leakage_conductances_g0 = nullptr;
      float * leak_reversal_potentials_E_L = nullptr;
      float * slope_factors_Delta_T = nullptr;
      float * adaptation_coupling_coefficients_a = nullptr;
      float * adaptation_time_constants_tau_w = nullptr;
      float * adaptation_changes_b = nullptr;

      ~AdExSpikingNeurons();
      MAKE_BACKEND_CONSTRUCTOR(AdExSpikingNeurons);
      using ::Backend::AdExSpikingNeurons::frontend;

      void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) override;
      void copy_constants_to_device() override;
      void check_for_neuron_spikes(float current_time_in_seconds, float timestep) override;
      void update_membrane_potentials(float timestep, float current_time_in_seconds) override;
      void reset_state() override;
    };

    __global__ void check_for_neuron_spikes_kernel
    (float *d_membrane_potentials_v,
     float *d_adaptation_values_w,
     float * d_adaptation_changes_b,
     float *d_thresholds_for_action_potential_spikes,
     float *d_resting_potentials,
     float* d_last_spike_time_of_each_neuron,
     unsigned char* d_bitarray_of_neuron_spikes,
     int bitarray_length,
     int bitarray_maximum_axonal_delay_in_timesteps,
     float current_time_in_seconds,
     float timestep,
     size_t total_number_of_neurons,
     bool high_fidelity_spike_flag);

    __global__ void AdEx_update_membrane_potentials
    (float *d_membrane_potentials_v,
     float * d_adaptation_values_w,
     float * d_adaptation_changes_b,
     float * d_membrane_capacitances_Cm,
     float * d_membrane_leakage_conductances_g0,
     float * d_leak_reversal_potentials_E_L,
     float * d_slope_factors_Delta_T,
     float * d_adaptation_coupling_coefficients_a,
     float * d_adaptation_time_constants_tau_w,
     float * d_current_injections,
     float * d_thresholds_for_action_potential_spikes,
     float * d_last_spike_time_of_each_neuron,
     float absolute_refractory_period,
     float current_time_in_seconds,
     float timestep,
     size_t total_number_of_neurons);
  }
}
