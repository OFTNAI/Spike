#pragma once

#include <cstddef>

#include "Spike/Base.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Macros.hpp"

class MemoryManager; // forward definition

namespace Backend {
  class MemoryManager : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(MemoryManager);

    void prepare() override {}
    void reset_state() override {}

    virtual std::size_t total_bytes() const = 0;
    virtual std::size_t free_bytes() const = 0;
  };
}

class MemoryManager : public virtual SpikeBase {
public:
  ~MemoryManager() override = default;
  SPIKE_ADD_BACKEND_GETSET(MemoryManager, SpikeBase);
  void init_backend(Context* ctx = _global_ctx) override;
  void reset_state() override;

  inline std::size_t backend_total_bytes() const {
    return backend()->total_bytes();
  }

  inline std::size_t backend_free_bytes() const {
    return backend()->free_bytes();
  }

private:
  std::shared_ptr<::Backend::MemoryManager> _backend;
};

namespace Backend {
  size_t memory_total_bytes(Context* ctx=nullptr);
  size_t memory_free_bytes(Context* ctx=nullptr);
}

inline void print_memory_usage() {
  float total_backend_bytes = Backend::memory_total_bytes();
  float free_backend_bytes = Backend::memory_free_bytes();
  float used_backend_bytes = total_backend_bytes - free_backend_bytes;

  printf("Backend memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_backend_bytes/1024.0/1024.0,
         free_backend_bytes/1024.0/1024.0,
         total_backend_bytes/1024.0/1024.0);
}
