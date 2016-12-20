#pragma once

#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/InputSpikingNeurons.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class SpikeAnalyser : public virtual ::Backend::SpikeAnalyser {
    public:
      MAKE_BACKEND_CONSTRUCTOR(SpikeAnalyser);
      using ::Backend::SpikeAnalyser::frontend;

      void prepare() override;
      void reset_state() override;

      void push_data_front() override;
      void pull_data_back() override;

      void store_spike_counts_for_stimulus_index(int stimulus_index) override;

    protected:
      ::Backend::CUDA::SpikingNeurons* neurons_backend = nullptr;
      ::Backend::CUDA::InputSpikingNeurons* input_neurons_backend = nullptr;
      ::Backend::CUDA::CountNeuronSpikesRecordingElectrodes* count_electrodes_backend = nullptr;
    };
  }
}