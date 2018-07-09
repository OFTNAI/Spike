#pragma once

#include "Spike/Plasticity/STDPPlasticity.hpp"
#include "Spike/Backend/CUDA/Plasticity/Plasticity.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    struct device_circular_spike_buffer_struct{
      int* time_buffer;
      int* id_buffer;
      int buffer_size;
    };
    class STDPPlasticity : public virtual ::Backend::CUDA::Plasticity,
        public virtual ::Backend::STDPPlasticity {
    protected:
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
      ::Backend::CUDA::SpikingSynapses* synapses_backend = nullptr;
    public:
      ~STDPPlasticity() override;
      using ::Backend::STDPPlasticity::frontend;
      int* plastic_synapse_indices = nullptr;
      int total_number_of_plastic_synapses;

      int* num_active_efferent_synapses = nullptr;
      int h_num_active_synapses;
      
      void prepare() override;
      void reset_state() override;
      void allocate_device_pointers();

    };
  }
}
