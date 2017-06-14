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

      ~AdExSpikingNeurons() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AdExSpikingNeurons);
      using ::Backend::AdExSpikingNeurons::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_to_device(); // Not virtual

      void state_update(float current_time_in_seconds, float timestep) override;
    };

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
     float * d_resting_potentials,
     float * d_last_spike_time_of_each_neuron,
     float absolute_refractory_period,
     float background_current,
     float current_time_in_seconds,
     float timestep,
     size_t total_number_of_neurons);
  }
}
