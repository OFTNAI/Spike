#pragma once
#include <cassert>
#include <iostream>

#ifndef NDEBUG
#include <cxxabi.h>
// From http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c#comment63837522_81870 :
#define TYPEID_NAME(x) abi::__cxa_demangle(typeid((x)).name(), NULL, NULL, NULL)
#endif

#define ADD_BACKEND_GETTER(TYPE)                \
  Backend::TYPE* backend() const {              \
    return (Backend::TYPE*)_backend;            \
  }

#define MAKE_PREPARE_BACKEND(TYPE) \
  void TYPE::prepare_backend(Context* ctx = _global_ctx) {      \
    std::cout << "prepare_backend " #TYPE " with " << ctx->device << "\n"; \
    switch (ctx->device) {                                      \
    case Backend::SPIKE_DEVICE_DUMMY:                           \
      _backend = new Backend::Dummy::TYPE();                    \
      break;                                                    \
    default:                                                    \
      assert("Unsupported backend" && false);                   \
    };                                                          \
    backend()->context = ctx;                                   \
    backend()->prepare();                                       \
    std::cout << "backend: " << _backend << "\n";               \
    std::cout << "this " #TYPE ": " << this << "\n";            \
  }

#define MAKE_STUB_PREPARE_BACKEND(TYPE)                         \
  void TYPE::prepare_backend(Context* ctx = _global_ctx) {      \
    assert("This type's backend cannot be instantiated!" && false);    \
  }
