#pragma once

#include "Spike/Helpers/RandomStateManager.hpp"

namespace Backend {
  namespace Dummy {
    class RandomStateManager : public ::Backend::RandomStateManager {
    public:
      virtual void prepare() {
        printf("TODO Backend::Dummy::RandomStateManager::prepare\n");
      }
    };
  }
}
