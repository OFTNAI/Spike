#pragma once

#include "Context.hpp"

#include <cassert>
#include <map>
#include <string>

namespace Backend {
  class SpikeBackendBase {
  public:
    bool ready = false;
    Context* context = _global_ctx;

    void* _frontend = nullptr;

    virtual ~SpikeBackendBase() = default;
    virtual void reset_state() = 0;
    virtual void prepare() = 0;
    virtual void push_data_front() = 0;
    virtual void pull_data_back() = 0;
  };

  template<typename FrontT, typename BackT>
  struct FactoryMap {
    typedef BackT* factory_t(FrontT*, Context*);
    typedef std::map<std::string, factory_t*> factory_map_t;
    factory_map_t map;
    inline factory_t*& operator[](const std::string& name) {
      factory_t*& factory = map[name];
      assert("Backend factory not registered!" && factory != nullptr);
      return factory;
    }
    inline factory_t*& operator[](std::string&& name) {
      return map[name];
    }
  };
} // namespace Backend

