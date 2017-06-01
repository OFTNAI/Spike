#pragma once

#include "Spike/Plasticity/WeightNormSpikingPlasticity.hpp"
#include "Plasticity.hpp"

#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class WeightNormSpikingPlasticity : public virtual ::Backend::CUDA::Plasticity,
                           public virtual ::Backend::WeightNormSpikingPlasticity {
    public:
      ~WeightNormSpikingPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(WeightNormSpikingPlasticity);
      using ::Backend::WeightNormSpikingPlasticity::frontend;

      int total_number_of_plastic_synapses = 0;
      int* plastic_synapse_indices = nullptr;

      float* sum_squared_afferent_values = nullptr;
      float* afferent_weight_change_updater = nullptr;

      bool* neuron_in_plasticity_set = nullptr;	
      float* initial_weights = nullptr;
      float* weight_divisor = nullptr;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      virtual void weight_normalization() override;

    protected:
      ::Backend::CUDA::Synapses* synapses_backend = nullptr;

    };

  // CUDA Kernel for calculating synapse changes
  __global__ void weight_change_calculations
	(int* postsyn_neuron,
	 float* current_weight,
	 float* initial_weights,
         float* afferent_weight_change_updater,
	 int* d_plastic_synapse_indices,
	 size_t total_number_of_plastic_synapses);


  // CUDA Kernel for determining the weight division
  __global__ void weight_division_calc
	(float* sum_squared_afferent_values,
	 float* afferent_weight_change_updater,
	 float* weight_divisor,
	 bool* neuron_in_plasticity_set,
	 size_t total_num_neurons);
 
  // Weight updating function
  __global__ void weight_update
	(int* postsyn_neuron,
	bool* neuron_in_plasticity_set,
	float* current_weight,
	float* weight_divisor,
	int* d_plastic_synapse_indices,
	size_t total_number_of_plastic_synapses); 

  }
}
