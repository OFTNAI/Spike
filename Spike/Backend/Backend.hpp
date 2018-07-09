#pragma once

#include "Context.hpp"
#include "Macros.hpp"

#include <cassert>
#include <map>
#include <string>

#ifndef NDEBUG
#include <iostream>
#endif

namespace Backend {
  class SpikeBackendBase {
  public:
    bool ready = false;
    Context* context = _global_ctx;

    void* _frontend = nullptr;

    virtual ~SpikeBackendBase() = default;
    virtual void reset_state() = 0;
    virtual void prepare() = 0;
  };

  template<typename FrontT, typename BackT>
  struct FactoryMap {
    typedef BackT* factory_t(FrontT*, Context*);
    typedef std::map<std::string, factory_t*> factory_map_t;
    factory_map_t map;
    inline factory_t*& operator[](const std::string& name) {
      factory_t*& factory = map[name];
      if (factory == nullptr) {
        factory = map["Dummy"];
#ifndef NDEBUG
        std::cerr << "Spike ERROR: Backend " << name
                  << " not registered for " << typeid(FrontT).name() << ".\n"
                  << "Spike ERROR: Falling back to Dummy..." << std::endl;
#endif
      }
      assert(factory != nullptr);
      return factory;
    }
    inline factory_t*& operator[](std::string&& name) {
      return map[name];
    }
  };
} // namespace Backend

