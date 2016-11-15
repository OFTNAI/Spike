#pragma once

#include "Spike/Neurons/Neurons.h"
#include "Spike/Backend/Dummy/DummyBackend.hpp"

namespace Backend {
  namespace Dummy {
    class Neurons : Generic {
    public:

      ::Neurons* front;
      
    };
  } // namespace Dummy
} // namespace Backend
