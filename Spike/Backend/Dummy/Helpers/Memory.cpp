#include "Memory.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, MemoryManager);

namespace Backend {
  namespace Dummy {
    void MemoryManager::prepare() {
    }

    std::size_t MemoryManager::total_bytes() const {
      return 0;
    }

    std::size_t MemoryManager::free_bytes() const {
      return 0;
    }
  }
}
