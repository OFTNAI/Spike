#ifndef TIMERWITHMESSAGES_H
#define TIMERWITHMESSAGES_H

#include <time.h>


class TimerWithMessages {

public:

	// Constructor/Destructor
	TimerWithMessages(const char * start_message);
	TimerWithMessages();

	clock_t clock_start;

	void stop_timer_and_log_time_and_message(const char * end_message, bool print_line_of_dashes);
	
};


#endif
