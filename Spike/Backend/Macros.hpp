#pragma once
#include <cassert>
#include <iostream>

#ifndef NDEBUG
#include <cxxabi.h>
// From http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c#comment63837522_81870 :
#define TYPEID_NAME(x) abi::__cxa_demangle(typeid((x)).name(), NULL, NULL, NULL)
#endif

#define ADD_BACKEND_GETTER(TYPE)                        \
  inline Backend::TYPE* backend() const {               \
    assert(_backend != nullptr &&                       \
           "Need to have backend initialized!");        \
    /*assert(((Backend::TYPE*)_backend)->ready &&   */  \
    /*       "Need to have backend ready!"); */         \
    return (Backend::TYPE*)_backend;                    \
  }                                                     \
  inline void prepare_backend() {                       \
    prepare_backend_early();                            \
    backend()->prepare();                               \
    prepare_backend_late();                             \
  }

#define MAKE_BACKEND_CONSTRUCTOR(TYPE)                  \
  TYPE(::TYPE* front) {                                 \
    std::cout << #TYPE " constructed with "             \
              << front << " as _frontend\n";            \
    _frontend = (void*)front;                           \
  }

#define ADD_FRONTEND_GETTER(TYPE)                       \
  inline ::TYPE* frontend() const {                     \
    assert(_frontend != nullptr &&                      \
           "Need to have backend initialized!");        \
    return (::TYPE*)_frontend;                          \
  }

#ifdef SPIKE_WITH_CUDA
#define MAKE_INIT_BACKEND(TYPE)                                         \
  void TYPE::init_backend(Context* ctx) {                               \
    std::cout << "init_backend " #TYPE " with " << ctx->device << "\n"; \
    switch (ctx->device) {                                              \
    case Backend::SPIKE_DEVICE_DUMMY:                                   \
      _backend = new Backend::Dummy::TYPE(this);                        \
      break;                                                            \
    case Backend::SPIKE_DEVICE_CUDA:                                    \
      _backend = new Backend::CUDA::TYPE(this);                         \
      break;                                                            \
    default:                                                            \
      assert("Unsupported backend" && false);                           \
    };                                                                  \
    backend()->context = ctx;                                           \
    prepare_backend();                                                  \
    std::cout << "backend: " << _backend << "\n";                       \
    std::cout << "this " #TYPE ": " << this << "\n";                    \
  }
#else
#define MAKE_INIT_BACKEND(TYPE)                                         \
  void TYPE::init_backend(Context* ctx) {                               \
    std::cout << "init_backend " #TYPE " with " << ctx->device << "\n"; \
    switch (ctx->device) {                                              \
    case Backend::SPIKE_DEVICE_DUMMY:                                   \
      _backend = new Backend::Dummy::TYPE(this);                        \
      break;                                                            \
    default:                                                            \
      assert("Unsupported backend" && false);                           \
    };                                                                  \
    backend()->context = ctx;                                           \
    prepare_backend();                                                  \
    std::cout << "backend: " << _backend << "\n";                       \
    std::cout << "this " #TYPE ": " << this << "\n";                    \
  }
#endif

#define MAKE_STUB_INIT_BACKEND(TYPE)                                   \
  void TYPE::init_backend(Context* ctx) {                              \
    assert("This type's backend cannot be instantiated!" && false);    \
  }
