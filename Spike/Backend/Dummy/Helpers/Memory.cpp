#include "Memory.hpp"

namespace Backend {
  namespace Dummy {
    std::size_t MemoryManager::total_bytes() const {
      return 0;
    }

    std::size_t MemoryManager::free_bytes() const {
      return 0;
    }
  }
}

SPIKE_EXPORT_BACKEND_TYPE(Dummy, MemoryManager);
