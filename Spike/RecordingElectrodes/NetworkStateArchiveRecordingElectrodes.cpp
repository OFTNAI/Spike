#include "NetworkStateArchiveRecordingElectrodes.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
//CUDA #include "../Helpers/CUDAErrorCheckHelpers.hpp"
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// NetworkStateArchiveRecordingElectrodes Constructor
NetworkStateArchiveRecordingElectrodes::NetworkStateArchiveRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param) 
	: RecordingElectrodes(neurons_parameter, synapses_parameter, full_directory_name_for_simulation_data_files_param, prefix_string_param) {

	network_state_archive_optional_parameters = new Network_State_Archive_Optional_Parameters();

}


// NetworkStateArchiveRecordingElectrodes Destructor
NetworkStateArchiveRecordingElectrodes::~NetworkStateArchiveRecordingElectrodes() {


}


void NetworkStateArchiveRecordingElectrodes::initialise_network_state_archive_recording_electrodes(Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters_param) {

	if (network_state_archive_optional_parameters_param != NULL) {
		network_state_archive_optional_parameters = network_state_archive_optional_parameters_param;
	}
	
}


void NetworkStateArchiveRecordingElectrodes::write_initial_synaptic_weights_to_file() {
	ofstream initweightfile;
	if (network_state_archive_optional_parameters->human_readable_storage){
		initweightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights_Initial.txt", ios::out | ios::binary);
		for (int i=0; i < synapses->total_number_of_synapses; i++){
			initweightfile << to_string(synapses->synaptic_efficacies_or_weights[i]) << endl;

		}
		initweightfile.close();
	} else {
		initweightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights_Initial.bin", ios::out | ios::binary);
		initweightfile.write((char *)synapses->synaptic_efficacies_or_weights, synapses->total_number_of_synapses*sizeof(float));
		initweightfile.close();
	}
}


void NetworkStateArchiveRecordingElectrodes::write_network_state_to_file() {

	clock_t save_network_state_start = clock();

	// Copy back the data that we might want:
	//CUDA CudaSafeCall(cudaMemcpy(synapses->synaptic_efficacies_or_weights, synapses->d_synaptic_efficacies_or_weights, sizeof(float)*synapses->total_number_of_synapses, cudaMemcpyDeviceToHost));
	
	if (network_state_archive_optional_parameters->human_readable_storage){
		// Creating and Opening all the files
		ofstream synapsepre, synapsepost, weightfile, delayfile;
		weightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights.txt", ios::out | ios::binary);
		delayfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkDelays.txt", ios::out | ios::binary);
		synapsepre.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPre.txt", ios::out | ios::binary);
		synapsepost.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPost.txt", ios::out | ios::binary);
		
		// Writing the data
		for (int i=0; i < synapses->total_number_of_synapses; i++){
			weightfile << to_string(synapses->synaptic_efficacies_or_weights[i]) << endl;
			delayfile << to_string(synapses->delays[i]) << endl;
			synapsepre << to_string(synapses->presynaptic_neuron_indices[i]) << endl;
			synapsepost << to_string(synapses->postsynaptic_neuron_indices[i]) << endl;
		}

		// Close files
		weightfile.close();
		delayfile.close();
		synapsepre.close();
		synapsepost.close();
	} else {
		// Creating and Opening all the files
		ofstream synapsepre, synapsepost, weightfile, delayfile;
		weightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights.bin", ios::out | ios::binary);
		delayfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkDelays.bin", ios::out | ios::binary);
		synapsepre.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPre.bin", ios::out | ios::binary);
		synapsepost.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPost.bin", ios::out | ios::binary);
		
		// Writing the data
		weightfile.write((char *)synapses->synaptic_efficacies_or_weights, synapses->total_number_of_synapses*sizeof(float));
		delayfile.write((char *)synapses->delays, synapses->total_number_of_synapses*sizeof(int));
		synapsepre.write((char *)synapses->presynaptic_neuron_indices, synapses->total_number_of_synapses*sizeof(int));
		synapsepost.write((char *)synapses->postsynaptic_neuron_indices, synapses->total_number_of_synapses*sizeof(int));

		// Close files
		weightfile.close();
		delayfile.close();
		synapsepre.close();
		synapsepost.close();
	}

	#ifndef QUIETSTART
	clock_t save_network_state_end = clock();
	float save_network_state_total_time = float(save_network_state_end - save_network_state_start) / CLOCKS_PER_SEC;
	printf("Network state saved to file.\n Time taken: %f\n", save_network_state_total_time);
	print_line_of_dashes_with_blank_lines_either_side();
	#endif

}
