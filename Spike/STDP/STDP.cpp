#include "STDP.hpp"

void STDP::reset_state() {
  backend()->reset_state();
}
