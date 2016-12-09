#pragma once

#include "Spike/Helpers/RandomStateManager.hpp"

namespace Backend {
  namespace Dummy {
    class RandomStateManager : public virtual ::Backend::RandomStateManager {
    public:
      MAKE_BACKEND_CONSTRUCTOR(RandomStateManager);

      void prepare() override {
        printf("TODO Backend::Dummy::RandomStateManager::prepare\n");
      }

      void push_data_front() override {}
      void pull_data_back() override {}
    };
  }
}
