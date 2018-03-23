// Define this to turn on error checking
#define PRINT_MESSAGES

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

inline void print_line_of_dashes_with_blank_lines_either_side(){
	printf("\n----------------------------------\n\n");
}


inline void print_message_and_exit(const char * message)
{
    printf("\n%s\nExiting...\n", message);

    exit(-1);
}

inline void check_for_epochs_and_begin_simulation_message(float timestep, int number_of_stimuli, int number_of_epochs, bool record_spikes, bool save_recorded_spikes_to_file, int total_number_of_neurons, int total_number_of_input_neurons, int total_number_of_synapses)
{
	if (number_of_epochs == 0) print_message_and_exit("Error. There must be at least one epoch.");

	#ifndef QUIETSTART

		printf("Beginning Simulation...\n");

		printf("Timestep: %f\nNumber of Stimuli: %d\nNumber of Epochs: %d\n", timestep, number_of_stimuli, number_of_epochs);

		printf("Total Number of Neurons: %d\n", total_number_of_neurons);
		printf("Total Number of Input Neurons: %d\n", total_number_of_input_neurons);
		printf("Total Number of Synapses: %d\n", total_number_of_synapses);
		
		if (record_spikes) printf("Spikes shall be recorded.\n");
		if ((record_spikes) && (save_recorded_spikes_to_file)) printf("Spikes shall be saved to file.\n");
		
		print_line_of_dashes_with_blank_lines_either_side();

	#endif
}

