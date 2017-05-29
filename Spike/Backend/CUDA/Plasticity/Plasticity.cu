// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/Plasticity.hpp"
#include <iostream>

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, Plasticity);

namespace Backend {
  namespace CUDA {
    Plasticity::~Plasticity() {
    }

    void Plasticity::prepare() {
      allocate_device_pointers();
    }

    void Plasticity::allocate_device_pointers(){
    }

    void Plasticity::reset_state() {
    }
  }
}
