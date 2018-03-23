// #include "catch.hpp"
// #include <iostream>
// #include <stdio.h>
// #include <stdlib.h>
// #include <fstream>
// #include <string>
// #include <sys/stat.h>
// // using namespace std;

// /**
// 		RECORDINGELECTRODES Test Set
// **/
// #include "Spike/RecordingElectrodes/RecordingElectrodes.hpp"
// #include "Spike/Neurons/SpikingNeurons.hpp"
// #include "Spike/Synapses/SpikingSynapses.hpp"
// TEST_CASE("RecordingElectrode") {
// 	SpikingSynapses test_synapses;
// 	SpikingNeurons test_neurons;

// 	// Creating the network
// 	// Pre-synaptic Population
// 	neuron_parameters_struct neuron_params_1;
// 	int dim1 = 1;
// 	int dim2 = 100;
// 	neuron_params_1.group_shape[0] = dim1;
// 	neuron_params_1.group_shape[1] = dim2;

// 	int presynaptic_population = test_neurons.AddGroup(&neuron_params_1);
	
// 	// Post-synaptic Population
// 	neuron_parameters_struct neuron_params_2;
// 	int dim1_2 = 1;
// 	int dim2_2 = 100;
// 	neuron_params_2.group_shape[0] = dim1_2;
// 	neuron_params_2.group_shape[1] = dim2_2;

// 	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);

// 	// Setting times
// 	float timestep = 0.01f;
// 	float current_time = 0.09f;

// 	// Setting Synapses up!
// 	spiking_synapse_parameters_struct synapse_params;
// 	synapse_params.stdp_on = true;
// 	synapse_params.delay_range[0] = 0.1f;
// 	synapse_params.delay_range[1] = 0.1f;
// 	// Set a fixed weight range
// 	synapse_params.weight_range_bottom = 0.5f;
// 	synapse_params.weight_range_top = 0.5f;
// 	// Connect
// 	synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
// 	test_synapses.AddGroup(
// 		presynaptic_population,
// 		postsynaptic_population,
// 		&test_neurons,
// 		&test_neurons,
// 		timestep,
// 		&synapse_params);


// 	mkdir("output/",S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH);


// 	// Checking for spikes on every timestep, with a store of length the number of neurons. Save spikes whenever they come.
// 	RecordingElectrodes test_record = RecordingElectrodes(&test_neurons, "output/", "test", 1, 1, 0.0f);

// 	// Setting up simulation
// 	test_neurons.set_threads_per_block_and_blocks_per_grid(512);
// 	test_synapses.set_threads_per_block_and_blocks_per_grid(512);
// 	// Allocating Pointers
// 	test_neurons.allocate_device_pointers(test_synapses.maximum_axonal_delay_in_timesteps, true);
// 	test_synapses.allocate_device_pointers();
// 	// Set-up Variables
// 	test_neurons.copy_constants_to_device();
// 	test_synapses.copy_constants_and_initial_efficacies_to_device();
// 	test_neurons.reset_neuron_activities();
// 	test_synapses.reset_synapse_activities();

// 	// Setting up the recording electrode
// 	test_record.allocate_pointers_for_spike_store();
// 	test_record.delete_and_reset_collected_spikes(); // From new CollectNeuronSpikesRecordingElectrodes. (RE Tests need updating for new classes)

// 	test_record.allocate_pointers_for_spike_count();
// 	test_record.reset_pointers_for_spike_count();

// 	SECTION("Testing Initial Weight Save"){
// 		// Saving the initial synaptic weights to file
// 		SECTION("Human Readable Initial Weight"){
// 			// Human readable form
// 			test_record.write_initial_synaptic_weights_to_file(&test_synapses, true);
// 			// Check the Results dir.
// 			std::ifstream weightfile;
// 			weightfile.open("./output/test_NetworkWeights_Initial.txt", std::ios::binary);
			
// 			// Check weights
// 			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
// 				std::string test_val;
// 				std::getline(weightfile, test_val);
// 				REQUIRE(std::stof(test_val) == test_synapses.synaptic_efficacies_or_weights[i]);
// 			}
// 			weightfile.close();
// 		}
// 		SECTION("Machine Readable Initial Weight"){
// 			// Non human readable form
// 			test_record.write_initial_synaptic_weights_to_file(&test_synapses, false);
// 			// Check the Results dir.
// 			std::ifstream weightfile;
// 			weightfile.open("./output/test_NetworkWeights_Initial.bin", std::ios::binary);
			
// 			// Check weights
// 			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
// 				float test_val;
// 				weightfile.read((char*)&test_val, sizeof(float));
// 				REQUIRE(test_val == test_synapses.synaptic_efficacies_or_weights[i]);
// 			}
// 			weightfile.close();
// 		}
// 	}

// 	// Set neuron last spike indices to those required:
// 	int indices[5] = {0, 12, 78, 9, 11};
// 	float* last_neuron_spike_times;
// 	last_neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
// 	CudaSafeCall(cudaMemcpy(last_neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
// 	// Set of the neuron spike times to now
// 	for (int i=0; i < 5; i++){
// 		last_neuron_spike_times[indices[i]] = current_time;
// 	}
// 	// Return the data to the device
// 	CudaSafeCall(cudaMemcpy(test_neurons.d_last_spike_time_of_each_neuron, last_neuron_spike_times, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));

// 	// Values will not necessarily be in order. So check that they are only checked once
// 	bool checked[5];
// 	for (int i=0; i < 5; i++){
// 		checked[i] = false;
// 	}

// 	SECTION("Spike Collect by timestep"){
// 		// Collect Spikes and check
// 		test_record.collect_spikes_for_timestep(current_time);
// 		// Copy spikes back and ensure that the values are correct
// 		int* number_of_spikes = (int*)malloc(sizeof(int));
// 		CudaSafeCall(cudaMemcpy(number_of_spikes, test_record.d_total_number_of_spikes_stored_on_device, sizeof(int), cudaMemcpyDeviceToHost));

// 		int* spiked_neuron_ids = (int*)malloc(sizeof(int)*number_of_spikes[0]);
// 		float* spiked_neuron_times = (float*)malloc(sizeof(float)*number_of_spikes[0]);
// 		CudaSafeCall(cudaMemcpy(spiked_neuron_ids, test_record.d_neuron_ids_of_stored_spikes_on_device, sizeof(int)*number_of_spikes[0], cudaMemcpyDeviceToHost));
// 		CudaSafeCall(cudaMemcpy(spiked_neuron_times, test_record.d_time_in_seconds_of_stored_spikes_on_device, sizeof(float)*number_of_spikes[0], cudaMemcpyDeviceToHost));

// 		// Check Values
// 		REQUIRE(number_of_spikes[0] == 5);
// 		for (int i=0; i < 5; i++){
// 			for (int j=0; j < 5; j++){
// 				if (spiked_neuron_ids[i] == indices[j]){
// 					if (checked[i] == false){
// 						checked[i] = true;
// 						REQUIRE(spiked_neuron_ids[i] == indices[j]);
// 						// printf("spiked_neuron_times[i]: %f\n", spiked_neuron_times[i]);
// 						REQUIRE(spiked_neuron_times[i] == current_time);
// 					} else {
// 						printf("Multiple copies of a single spike!");
// 						REQUIRE(true == false);
// 					}
// 				}
// 			}
// 		}
// 	}

// 	SECTION("Testing method to copy spiked neuron ids & times"){
// 		// Collect Spikes and check
// 		test_record.collect_spikes_for_timestep(current_time);
// 		// Test the effect when we use the spike copy function
// 		test_record.copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time, 0, 1);
// 		REQUIRE(test_record.h_total_number_of_spikes_stored_on_host == 5);

// 		for (int i=0; i < 5; i++){
// 			for (int j=0; j < 5; j++){
// 				if (test_record.h_neuron_ids_of_stored_spikes_on_host[i] == indices[j]){
// 					if (checked[i] == false){
// 						checked[i] = true;
// 						REQUIRE(test_record.h_neuron_ids_of_stored_spikes_on_host[i] == indices[j]);
// 						REQUIRE(test_record.h_time_in_seconds_of_stored_spikes_on_host[i] == current_time);
// 					} else {
// 						printf("Multiple copies of a single spike!");
// 						REQUIRE(true == false);
// 					}
// 				}
// 			}
// 		}
// 	}

// 	SECTION("Adding to per neuron spike count"){
// 		test_record.add_spikes_to_per_neuron_spike_count(current_time);
// 		int* spikes_per_neuron = (int*)malloc(sizeof(int)*test_neurons.total_number_of_neurons);
// 		CudaSafeCall(cudaMemcpy(spikes_per_neuron, test_record.d_per_neuron_spike_counts, sizeof(int)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
// 		for (int i=0; i < test_neurons.total_number_of_neurons; i++){
// 			if ((i == indices[0]) || (i == indices[1]) || (i == indices[2]) || (i == indices[3]) || (i == indices[4])){
// 				REQUIRE(spikes_per_neuron[i] == 1);
// 			} else {
// 				REQUIRE(spikes_per_neuron[i] == 0);
// 			}
// 		}
// 	}

// 	SECTION("Write Spikes to file"){
// 		// Collect Spikes
// 		test_record.collect_spikes_for_timestep(current_time);
// 		test_record.copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time, 0, 1);
		
// 		SECTION("Human Readable Spike Write Test"){
// 			// Save the spikes to file
// 			test_record.write_spikes_to_file(0, true, false);
// 			// Open the saved spikes and check contents
// 			std::ifstream savedspikeids;
// 			std::ifstream savedspiketimes;
// 			savedspikeids.open("./output/test_SpikeIDs_Untrained_Epoch0.txt", std::ios::binary);
// 			savedspiketimes.open("./output/test_SpikeTimes_Untrained_Epoch0.txt", std::ios::binary);
			
// 			// Check values
// 			for (int i=0; i < 5; i++){
// 				std::string test_id;
// 				std::getline(savedspikeids, test_id);
// 				for (int j=0; j < 5; j++){
// 					if (std::stoi(test_id) == indices[j]){
// 						if (checked[i] == false){
// 							std::string test_spike;
// 							std::getline(savedspiketimes, test_spike);
// 							checked[i] = true;
// 							REQUIRE(std::stoi(test_id) == indices[j]);
// 							REQUIRE(std::stof(test_spike) == current_time);
// 						} else {
// 							printf("Multiple copies of a single spike!");
// 							REQUIRE(true == false);
// 						}
// 					}
// 				}
// 			}
// 			savedspiketimes.close();
// 			savedspikeids.close();
// 		}
// 		SECTION("Machine Readable Spike Write Test"){
// 			// Save the spikes to file
// 			test_record.write_spikes_to_file(0, false, false);
// 			// Open the saved spikes and check contents
// 			std::ifstream savedspikeids;
// 			std::ifstream savedspiketimes;
// 			savedspikeids.open("./output/test_SpikeIDs_Untrained_Epoch0.bin", std::ios::binary);
// 			savedspiketimes.open("./output/test_SpikeTimes_Untrained_Epoch0.bin", std::ios::binary);
			
// 			// Check values
// 			for (int i=0; i < 5; i++){
// 				int test_id;
// 				savedspikeids.read((char*)&test_id, sizeof(int));
// 				for (int j=0; j < 5; j++){
// 					if (test_id == indices[j]){
// 						if (checked[i] == false){
// 							float test_spike;
// 							savedspiketimes.read((char*)&test_spike, sizeof(float));
// 							checked[i] = true;
// 							REQUIRE(test_id == indices[j]);
// 							REQUIRE(test_spike == current_time);
// 						} else {
// 							printf("Multiple copies of a single spike!");
// 							REQUIRE(true == false);
// 						}
// 					}
// 				}
// 			}
// 			savedspiketimes.close();
// 			savedspikeids.close();
// 		}
// 	}

// 	SECTION("Save Network State"){
// 		SECTION("Human Readable Network State Storage"){
// 			// Save the network to files
// 			test_record.write_network_state_to_file(&test_synapses, true);
// 			// Open the various file outputs and check the values

// 			std::ifstream weightfile, delayfile, prefile, postfile;
// 			weightfile.open("./output/test_NetworkWeights.txt", std::ios::binary);
// 			delayfile.open("./output/test_NetworkDelays.txt", std::ios::binary);
// 			prefile.open("./output/test_NetworkPre.txt", std::ios::binary);
// 			postfile.open("./output/test_NetworkPost.txt", std::ios::binary);
			
// 			// Check weights
// 			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
// 				std::string test_weight, test_delay, test_pre, test_post;
// 				std::getline(weightfile, test_weight);
// 				std::getline(delayfile, test_delay);
// 				std::getline(prefile, test_pre);
// 				std::getline(postfile, test_post);
// 				REQUIRE(std::stof(test_weight) == test_synapses.synaptic_efficacies_or_weights[i]);
// 				REQUIRE(std::stoi(test_delay) == test_synapses.delays[i]);
// 				REQUIRE(std::stoi(test_pre) == test_synapses.presynaptic_neuron_indices[i]);
// 				REQUIRE(std::stoi(test_post) == test_synapses.postsynaptic_neuron_indices[i]);
// 			}
// 			weightfile.close();
// 			delayfile.close();
// 			prefile.close();
// 			postfile.close();
// 		}
// 		SECTION("Machine Readable Network State Storage"){
// 			// Save the network to files
// 			test_record.write_network_state_to_file(&test_synapses, false);
// 			// Open the various file outputs and check the values

// 			std::ifstream weightfile, delayfile, prefile, postfile;
// 			weightfile.open("./output/test_NetworkWeights.bin", std::ios::binary);
// 			delayfile.open("./output/test_NetworkDelays.bin", std::ios::binary);
// 			prefile.open("./output/test_NetworkPre.bin", std::ios::binary);
// 			postfile.open("./output/test_NetworkPost.bin", std::ios::binary);
			
// 			// Check weights
// 			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
// 				float test_weight;
// 				int test_delay, test_pre, test_post;
// 				weightfile.read((char*)&test_weight, sizeof(float));
// 				delayfile.read((char*)&test_delay, sizeof(int));
// 				prefile.read((char*)&test_pre, sizeof(int));
// 				postfile.read((char*)&test_post, sizeof(int));
// 				REQUIRE(test_weight == test_synapses.synaptic_efficacies_or_weights[i]);
// 				REQUIRE(test_delay == test_synapses.delays[i]);
// 				REQUIRE(test_pre == test_synapses.presynaptic_neuron_indices[i]);
// 				REQUIRE(test_post == test_synapses.postsynaptic_neuron_indices[i]);
// 			}
// 			weightfile.close();
// 			delayfile.close();
// 			prefile.close();
// 			postfile.close();
// 		}
// 	}
// }
