#include "TimerWithMessages.hpp"
#include <iostream>
#include "TerminalHelpers.hpp"

// TimerWithMessages Constructor
TimerWithMessages::TimerWithMessages(const char * start_message) {
  clock_start = clock();
  printf("%s", start_message);
}

TimerWithMessages::TimerWithMessages() {
  clock_start = clock();
}

void TimerWithMessages::stop_timer_and_log_time_and_message(const char * end_message, bool print_line_of_dashes) {

  clock_t clock_end = clock();

  float time_elapsed = float(clock_end - clock_start) / CLOCKS_PER_SEC;

  printf("%s Time taken: %f\n", end_message, time_elapsed);

  if (print_line_of_dashes)
    print_line_of_dashes_with_blank_lines_either_side();

}
