#pragma once

#include "Spike/Neurons/InputSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"
#include "Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.hpp"
#include "Spike/Backend/CUDA/Synapses/VoltageSpikingSynapses.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class InputSpikingNeurons : public virtual ::Backend::CUDA::SpikingNeurons,
                                public virtual ::Backend::InputSpikingNeurons {
    public:
      void prepare() override;
      void reset_state() override;
    };
  }
}
