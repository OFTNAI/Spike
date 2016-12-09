#include "RandomStateManager.hpp"

void RandomStateManager::reset_state() {
  backend()->reset_state();
}

MAKE_PREPARE_BACKEND(RandomStateManager);
