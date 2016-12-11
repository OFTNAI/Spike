#include "RandomStateManager.hpp"

void RandomStateManager::reset_state() {
  backend()->reset_state();
}

MAKE_INIT_BACKEND(RandomStateManager);
