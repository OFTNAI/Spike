#include "PatternedPoissonInputSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <algorithm> // For random shuffle

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "../Helpers/FstreamWrapper.hpp"

using namespace std;


PatternedPoissonInputSpikingNeurons::PatternedPoissonInputSpikingNeurons() {
  stimuli_rates = nullptr;
  total_number_of_rates = 0;
}

PatternedPoissonInputSpikingNeurons::~PatternedPoissonInputSpikingNeurons() {
  free(stimuli_rates);
}

void PatternedPoissonInputSpikingNeurons::state_update
(float current_time_in_seconds, float timestep) {
  backend()->state_update(current_time_in_seconds, timestep);
}

void PatternedPoissonInputSpikingNeurons::reset_stimuli(){
  total_number_of_rates = 0;
  total_number_of_input_stimuli = 0;
  free(stimuli_rates);
  stimuli_rates = nullptr;
}

int PatternedPoissonInputSpikingNeurons::add_stimulus(float* rates, int num_rates){
  // Check if the size of the rates is correct
  if (num_rates != total_number_of_neurons){
    printf("Error: The number of neurons does not match the number of rates!\n");
    exit(1);
  }
  // If correct, allocate some memory save these values
  stimuli_rates = (float*)realloc(stimuli_rates, sizeof(float)*(total_number_of_rates + num_rates));
  for (int index = 0; index < num_rates; index++){
    stimuli_rates[total_number_of_rates + index] = rates[index];
  }

  // After storing the stimulus, correct the number of input stimuli
  total_number_of_rates += num_rates;
  ++total_number_of_input_stimuli;

  return(total_number_of_input_stimuli - 1);
}

int PatternedPoissonInputSpikingNeurons::add_stimulus(std::vector<float> rates){
  return(add_stimulus(rates.data(), rates.size()));
}

void PatternedPoissonInputSpikingNeurons::copy_rates_to_device() {
  backend()->copy_rates_to_device();
}

SPIKE_MAKE_INIT_BACKEND(PatternedPoissonInputSpikingNeurons);

