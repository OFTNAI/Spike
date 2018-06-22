#include "SpikeMonitors.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// SpikeMonitors Constructor
SpikeMonitors::SpikeMonitors(SpikingNeurons * neurons_parameter) : Monitors(neurons_parameter) {

  // Variables
  size_of_device_spike_store = 0;
  total_number_of_spikes_stored_on_host = 0;

  // Host Pointers
  advanced_parameters = new spike_monitor_advanced_parameters();
  neuron_ids_of_stored_spikes_on_host = nullptr;
  spike_times_of_stored_spikes_on_host = nullptr;
  total_number_of_spikes_stored_on_device = nullptr;

  // Private Host Pointeres
  reset_neuron_ids = nullptr;
  reset_neuron_times = nullptr;

}


// SpikeMonitors Destructor
SpikeMonitors::~SpikeMonitors() {
  free(neuron_ids_of_stored_spikes_on_host);
  free(spike_times_of_stored_spikes_on_host);
  free(total_number_of_spikes_stored_on_device);

  free(reset_neuron_ids);
  free(reset_neuron_times);
}


void SpikeMonitors::prepare_backend_early() {

  size_of_device_spike_store = advanced_parameters->device_spike_store_size_multiple_of_total_neurons * neurons->total_number_of_neurons;
  allocate_pointers_for_spike_store();

}



void SpikeMonitors::allocate_pointers_for_spike_store() {

  total_number_of_spikes_stored_on_device = (int*)malloc(sizeof(int));
  total_number_of_spikes_stored_on_device[0] = 0;

  reset_neuron_ids = (int *)malloc(sizeof(int)*size_of_device_spike_store);
  reset_neuron_times = (float *)malloc(sizeof(float)*size_of_device_spike_store);
  for (int i=0; i < size_of_device_spike_store; i++){
    reset_neuron_ids[i] = -1;
    reset_neuron_times[i] = -1.0f;
  }
}

void SpikeMonitors::reset_state() {

  // Reset the spike store
  // Host values
  total_number_of_spikes_stored_on_host = 0;
  total_number_of_spikes_stored_on_device[0] = 0;
  // Free/Clear Device stuff
  // Reset the number on the device
  backend()->reset_state();

  // Free malloced host stuff
  free(neuron_ids_of_stored_spikes_on_host);
  free(spike_times_of_stored_spikes_on_host);
  neuron_ids_of_stored_spikes_on_host = nullptr;
  spike_times_of_stored_spikes_on_host = nullptr;
}


void SpikeMonitors::copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, float timestep, bool force) {
  int current_time_in_timesteps = round(current_time_in_timesteps / timestep);

  if (((current_time_in_timesteps % advanced_parameters->number_of_timesteps_per_device_spike_copy_check) == 0) || force){

    // Finally, we want to get the spikes back. Every few timesteps check the number of spikes:
    backend()->copy_spikecount_to_front();

    // Ensure that we don't have too many
    if (total_number_of_spikes_stored_on_device[0] > size_of_device_spike_store){
      print_message_and_exit("Spike recorder has been overloaded! Reduce threshold.");
    }

    // Deal with them!
    if ((total_number_of_spikes_stored_on_device[0] >= (advanced_parameters->proportion_of_device_spike_store_full_before_copy * size_of_device_spike_store)) ||  force){

      // Reallocate host spike arrays to accommodate for new device spikes.
      neuron_ids_of_stored_spikes_on_host = (int*)realloc(neuron_ids_of_stored_spikes_on_host, sizeof(int)*(total_number_of_spikes_stored_on_host + total_number_of_spikes_stored_on_device[0]));
      spike_times_of_stored_spikes_on_host = (float*)realloc(spike_times_of_stored_spikes_on_host, sizeof(float)*(total_number_of_spikes_stored_on_host + total_number_of_spikes_stored_on_device[0]));

      // Copy device spikes into correct host array location
      backend()->copy_spikes_to_front();

      total_number_of_spikes_stored_on_host += total_number_of_spikes_stored_on_device[0];


      // Reset device spikes
      backend()->reset_state();
      total_number_of_spikes_stored_on_device[0] = 0;
    }
  }
}


void SpikeMonitors::state_update(float current_time_in_seconds, float timestep){
  copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time_in_seconds, timestep);
}

void SpikeMonitors::final_update(float current_time_in_seconds, float timestep){
  copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time_in_seconds, timestep, true);
}


SPIKE_MAKE_INIT_BACKEND(SpikeMonitors);
