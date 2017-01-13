#pragma once

#include "Spike/Helpers/Memory.hpp"

namespace Backend {
  namespace CUDA {
    class MemoryManager : public virtual ::Backend::MemoryManager {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(MemoryManager);

      void prepare() override {
      }

      std::size_t total_bytes() const override;
      std::size_t free_bytes() const override;
    };
  }
}
