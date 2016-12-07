#ifndef RANDOMSTATEMANAGER_H
#define RANDOMSTATEMANAGER_H

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

class RandomStateManager; // forward definition

namespace Backend {
  class RandomStateManager : public SpikeBackendBase {
  public:
    ADD_FRONTEND_GETTER(RandomStateManager);
    void reset_state() override {
      prepare();
    }

    void push_data_front() override {} // TODO
    void pull_data_back() override {} // TODO
  };
}

#include "Spike/Backend/Dummy/RandomStateManager.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/RandomStateManager.hpp"
#endif


class RandomStateManager : public virtual SpikeBase {
public:
  int total_number_of_states = 0;

  ADD_BACKEND_GETTER(RandomStateManager);
  void prepare_backend(Context* ctx = _global_ctx);
  virtual void reset_state() {}
private:
  static RandomStateManager *inst;
};

#endif
