#ifndef RANDOMSTATEMANAGER_H
#define RANDOMSTATEMANAGER_H

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

class RandomStateManager; // forward definition

namespace Backend {
  // No `RandomStateManagerCommon' necessary, I believe
  class RandomStateManager : public SpikeBackendBase {
  public:
    virtual void reset_state() {
      prepare();
    }
  };
}

#include "Spike/Backend/Dummy/RandomStateManager.hpp" 

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
