#pragma once

#include "Spike/Backend/Device.hpp"
#include "Spike/Backend/Backend.hpp"

namespace Backend {
  namespace CUDA {
    class Device : ::Backend::Device
    {
    }; // ::Backend::CUDA::Device
    
    class Generic : ::Backend::Generic
    {
    }; // ::Backend::CUDA::Generic
  } // namespace CUDA
}
