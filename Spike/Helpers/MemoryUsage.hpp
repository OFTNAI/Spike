#pragma once

#include <cstddef>

#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Memory.hpp"

inline void print_memory_usage() {
  const Context* ctx = Backend::get_current_context();
  float total_backend_bytes = Backend::memory_total_bytes();
  float free_backend_bytes = Backend::memory_free_bytes();
  float used_backend_bytes = total_backend_bytes - free_backend_bytes;

  printf("Backend memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_backend_bytes/1024.0/1024.0,
         free_backend_bytes/1024.0/1024.0,
         total_backend_bytes/1024.0/1024.0);
}
