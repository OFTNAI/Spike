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
    ADD_FRONTEND_GETTER(RandomStateManager);

    void reset_state() override {
      // Unorthodox: reset_state doesn't usually just mean, 'call prepare()'
      prepare();
    }
    void push_data_front() override {}
    void pull_data_back() override {}
  };
}

#include "Spike/Backend/Dummy/Helpers/RandomStateManager.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Helpers/RandomStateManager.hpp"
#endif


class RandomStateManager : public virtual SpikeBase {
public:
  ADD_BACKEND_GETSET(RandomStateManager, SpikeBase);
  void init_backend(Context* ctx = _global_ctx) override;
  void reset_state() override;
private:
  static RandomStateManager *inst;
  ::Backend::RandomStateManager* _backend = nullptr;
};

#endif