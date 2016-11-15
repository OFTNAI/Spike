#pragma once

#include "Spike/Neurons/Neurons.hpp"
#include "Spike/Backend/Dummy/DummyBackend.hpp"

namespace Backend {
  namespace Dummy {
    class Neurons : Generic {
    public:

      ::Neurons* front;
      
    };
  } // namespace Dummy
} // namespace Backend
