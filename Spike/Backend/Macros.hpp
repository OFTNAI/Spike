#pragma once
#include <cassert>
#include <iostream>

#define MAKE_PREPARE_BACKEND(TYPE) \
  void TYPE::prepare_backend(Context* ctx = _global_ctx) {      \
    std::cout << "prepare_backend " #TYPE " with " << ctx->device << "\n"; \
    switch (ctx->device) {                                      \
    case Backend::SPIKE_DEVICE_DUMMY:                           \
      backend = new Backend::Dummy::TYPE();                     \
      break;                                                    \
    default:                                                    \
      assert("Unsupported backend" && false);                   \
    };                                                          \
    backend->context = ctx;                                     \
    backend->prepare();                                         \
    std::cout << "backend: " << backend << "\n";                \
  }

#define MAKE_STUB_PREPARE_BACKEND(TYPE)                         \
  void TYPE::prepare_backend(Context* ctx = _global_ctx) {      \
    assert("This type shouldn't be instantiated!" && false);    \
  }
