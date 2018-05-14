#pragma once

#include "Spike/Plasticity/WeightDependentSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class WeightDependentSTDPPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                           public virtual ::Backend::WeightDependentSTDPPlasticity {
    public:
      float* stdp_pre_memory_trace = nullptr;
      float* stdp_pre_memory_trace_update_time = nullptr;
      float* stdp_post_memory_trace = nullptr;
      float* h_stdp_trace = nullptr;
      float* weight_update_vals = nullptr;
      float* weight_update_times = nullptr;

      int* num_activated_post_neurons = nullptr;
      int* activated_post_neuron_ids = nullptr;
      int* active_afferent_synapse_counts = nullptr;
      int* num_active_afferent_synapses = nullptr;
      int h_num_active_afferent_synapses;

      ~WeightDependentSTDPPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(WeightDependentSTDPPlasticity);
      using ::Backend::WeightDependentSTDPPlasticity::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) override;
    };
    __global__ void post_trace_update(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        int* post_neuron_set,
        float* d_last_spike_time_of_each_neuron,
        float* stdp_post_memory_trace,
        int* num_activated_post_neurons,
        int* activated_post_neuron_ids,
        int* num_active_afferent_synapses,
        int* post_neuron_afferent_counts,
        float current_time_in_seconds,
        float timestep,
        size_t num_post_neurons);
    __global__ void pre_trace_update(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        device_circular_spike_buffer_struct spike_buffer,
        int bufferloc,
        int total_number_of_plastic_synapses,
        float* stdp_pre_memory_trace,
        float* stdp_pre_memory_trace_update_time,
        float current_time_in_seconds,
        float timestep);
    
    __global__ void on_pre(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        device_circular_spike_buffer_struct spike_buffer,
        int bufferloc,
        int total_number_of_plastic_synapses,
        int* postsynaptic_neuron_ids,
        int* post_neuron_conversion,
        int* plastic_synapses,
        float* stdp_post_memory_trace,
        float* synaptic_efficacies_or_weights,
        float* weight_update_vals,
        float* weight_update_times,
        float current_time_in_seconds,
        float timestep);
    __global__ void on_post(
        weightdependent_stdp_plasticity_parameters_struct stdp_vars,
        int* activated_post_neuron_ids,
        int* post_neuron_afferent_counts,
        int* post_neuron_afferent_totals,
        int* post_neuron_afferent_ids,
        int* num_active_afferent_synapses,
        int total_number_of_plastic_synapses,
        int* plastic_synapses,
        float* stdp_pre_memory_trace,
        float* synaptic_efficacies_or_weights,
        float* weight_update_vals,
        float* weight_update_times,
        float current_time_in_seconds,
        float timestep);
  /*
    __global__ void ltp_and_ltd
          (int* d_postsyns,
           float* d_time_of_last_spike_to_reach_synapse,
           float* d_last_spike_time_of_each_neuron,
           float* d_synaptic_efficacies_or_weights,
           float* stdp_pre_memory_trace,
           float* stdp_post_memory_trace,
           struct weightdependent_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           float current_time_in_seconds,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses);
           */

  }
}
