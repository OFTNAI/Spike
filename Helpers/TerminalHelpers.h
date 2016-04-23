// Define this to turn on error checking
#define PRINT_MESSAGES

#include <stdio.h>
#include <iostream>
#include <stdlib.h>


inline void print_message_and_exit(const char * message)
{
    
    // #ifdef PRINT_MESSAGES
        // printf("JI PRINT MESSAGES\n");
    // #endif

	printf("%s\n", message);
    printf("Exiting...\n");
    exit(-1);

    return;
}
