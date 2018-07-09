#include "RandomStateManager.hpp"

void RandomStateManager::reset_state() {
  backend()->reset_state();
}

SPIKE_MAKE_INIT_BACKEND(RandomStateManager);
