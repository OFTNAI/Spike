#ifndef RANDOMSTATEMANAGER_H
#define RANDOMSTATEMANAGER_H

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

class RandomStateManager; // forward definition

namespace Backend {
  class RandomStateManager : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(RandomStateManager);

    void reset_state() override {
      // Unorthodox: reset_state doesn't usually just mean, 'call prepare()'
      prepare();
    }
  };
}

class RandomStateManager : public virtual SpikeBase {
public:
  ~RandomStateManager() override = default;
  SPIKE_ADD_BACKEND_GETSET(RandomStateManager, SpikeBase);
  void init_backend(Context* ctx = _global_ctx) override;
  void reset_state() override;
private:
  static RandomStateManager *inst;
  std::shared_ptr<::Backend::RandomStateManager> _backend;
};

#endif
