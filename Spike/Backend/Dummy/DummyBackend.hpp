#pragma once

#include "Spike/Backend/Device.hpp"
#include "Spike/Backend/Backend.h"

namespace Backend {
  namespace Dummy {
    class Device : ::Backend::Device
    {
    }; // ::Backend::Dummy::Device
    
    class Generic : ::Backend::Generic
    {
    }; // ::Backend::Dummy::Generic
  } // namespace Dummy
} // namespace Backend
