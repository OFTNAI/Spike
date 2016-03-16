// Creating a set of tests for my Spike Simulator
// Using Catch.hpp to carry out testing
//
//	Author: Nasir Ahmad
//	Date of Creation: 16/12/2015

#define CATCH_CONFIG_MAIN
#define GPU_TEST
#include "catch.hpp"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;
#include "Structs.h"

/*

		NEURON CLASS TESTS

*/

// Unit testing
#include "NeuronPopulations.h"
// First testing the constructor
TEST_CASE("Neuron Class Constructor", "[Neuron]") {
	// Create an instance of Neuron
	NeuronPopulations neurtest;
	// Test the default settings
	REQUIRE(neurtest.numNeurons == 0);
	REQUIRE(neurtest.numPopulations == 0);
	REQUIRE(neurtest.numperPop == NULL);
}

// Next test the creation of a population
TEST_CASE("Neuron Class AddConnection", "[Neuron]") {
	// Create an instance of Neuron
	NeuronPopulations neurtest;
	//float params[] = {0.01f, 0.02f, 0.03f, 0.04f};
	struct neuron_struct params;
	params.parama = 0.01f;
	params.paramb = 0.02f;
	params.paramc = 0.03f;
	params.paramd = 0.04f;
	// TESTING SINGLE POPULATION
	neurtest.AddPopulation(100, params);
	// Check that the number of neurons has been correctly set
	REQUIRE(neurtest.numNeurons == 100);
	REQUIRE(neurtest.numPopulations == 1);
	REQUIRE(neurtest.numperPop[0] == neurtest.numNeurons);
	// Check that the values in every part of array
	for (int i = 0; i < neurtest.numNeurons; i++){
		// Set the parameters
		REQUIRE(neurtest.neuronpop_variables[i].parama == 0.01f);
		REQUIRE(neurtest.neuronpop_variables[i].paramb == 0.02f);
		REQUIRE(neurtest.neuronpop_variables[i].paramc == 0.03f);
		REQUIRE(neurtest.neuronpop_variables[i].paramd == 0.04f);
		// Set state variables
		REQUIRE(neurtest.neuronpop_variables[i].state_v == -70.0f);
		REQUIRE(neurtest.neuronpop_variables[i].state_u == 0.0f);
	}
	// TESTING MULTIPLE POPULATIONS
	struct neuron_struct paramstwo;
	paramstwo.parama = 0.04f;
	paramstwo.paramb = 0.03f;
	paramstwo.paramc = 0.02f;
	paramstwo.paramd = 0.01f;
	neurtest.AddPopulation(28,paramstwo);
	// Check that the number of neurons has been correctly set
	REQUIRE(neurtest.numNeurons == 128);
	REQUIRE(neurtest.numPopulations == 2);
	REQUIRE(neurtest.numperPop[0] == 100);
	REQUIRE(neurtest.numperPop[1] == neurtest.numNeurons);
	// Check that the values in every part of array
	for (int i = 0; i < neurtest.numNeurons; i++){
		if (i < 100){
			// Set the parameters
			REQUIRE(neurtest.neuronpop_variables[i].parama == 0.01f);
			REQUIRE(neurtest.neuronpop_variables[i].paramb == 0.02f);
			REQUIRE(neurtest.neuronpop_variables[i].paramc == 0.03f);
			REQUIRE(neurtest.neuronpop_variables[i].paramd == 0.04f);
			// Set state variables
			REQUIRE(neurtest.neuronpop_variables[i].state_v == -70.0f);
			REQUIRE(neurtest.neuronpop_variables[i].state_u == 0.0f);
		} else {
			REQUIRE(neurtest.neuronpop_variables[i].parama == 0.04f);
			REQUIRE(neurtest.neuronpop_variables[i].paramb == 0.03f);
			REQUIRE(neurtest.neuronpop_variables[i].paramc == 0.02f);
			REQUIRE(neurtest.neuronpop_variables[i].paramd == 0.01f);
			// Set state variables
			REQUIRE(neurtest.neuronpop_variables[i].state_v == -70.0f);
			REQUIRE(neurtest.neuronpop_variables[i].state_u == 0.0f);
		}
	}
}









/*

		SYNAPSE CLASS TESTS

*/

// Unit testing
#include "Synapse.h"
// First testing construction
TEST_CASE("Synapse Class Constructor", "[Synapse]"){
	// Create an instance of Synapse
	Synapse syn;
	// Testing initial vals
	REQUIRE(syn.numconnections == 0);
	REQUIRE(syn.w_max == 60.0f);
	REQUIRE(syn.a_minus == -0.015f);
	REQUIRE(syn.a_plus == 0.005f);
	REQUIRE(syn.tau_minus == 0.025f);
	REQUIRE(syn.tau_plus == 0.015f);
	// After setting STDP check that the values change
	syn.SetSTDP(0.01f, 0.02f, 0.03f, 0.04f, 0.05f);
	REQUIRE(syn.w_max == 0.01f);
	REQUIRE(syn.a_minus == 0.02f);
	REQUIRE(syn.a_plus == 0.03f);
	REQUIRE(syn.tau_minus == 0.04f);
	REQUIRE(syn.tau_plus == 0.05f);
}
// Next testing what happens when we create a set of Synapses SOLO
TEST_CASE("Synapse ALL_TO_ALL SOLO Creation", "[Synapse]"){
	// Create an instance of Synapse
	Synapse syn;
	// Setting fictional pre and post IDs
	int pre = 0;
	int post = 1;
	// Setting their population numbers
	int popNums[] = {100, 250};
	// Setting up the connection style
	char style[] = "all_to_all";
	// Setting up Weights
	float weightrange[] = {1.0f, 1.0f};
	// Setting up Delays
	int delayrange[] = {1, 1};
	// Determining whether or not there is STDP
	bool stdpswitch = false;
	// Giving the parameter
	float parameter = 0.0f;
	// Creating the Synapses
	syn.AddConnection(pre, 
						post, 
						popNums, 
						style, 
						weightrange,
						delayrange,
						stdpswitch,
						parameter);
	// Time to test
	for (int i=0; i < popNums[0]; i++){
		for (int j=popNums[0]; j < popNums[1]; j++){
			// The index is the value of i + (j-initj)*max_i
			int index = i + (j-popNums[0])*popNums[0];
			// Check the connections
			REQUIRE(syn.presyns[index] == i);
			REQUIRE(syn.postsyns[index] == j);
			REQUIRE(syn.weights[index] == 1.0f);
			REQUIRE(syn.delays[index] == 1.0f);
			REQUIRE(syn.stdp[index] == 0);
		}
	}
	// Test the number of connections
	REQUIRE(syn.numconnections == popNums[0]*(popNums[1] - popNums[0]));

	//////////// RANDOM WEIGHTS AND DELAYS ////////////
	// Creating a new set of ranges
	float weightrange_rand[] = {1.0f, 25.0f};
	int delayrange_rand[] = {2, 60};
	// Calculating the specific average of these ranges
	float weightrangeavg = std::abs((weightrange_rand[1] + weightrange_rand[0])/2.0f);
	float delayrangeavg = std::abs((float)(delayrange_rand[1] + delayrange_rand[0])/2.0f);
	// Creating the Synapses
	syn.AddConnection(pre, 
						post, 
						popNums, 
						style, 
						weightrange_rand,
						delayrange_rand,
						stdpswitch,
						parameter);
	// Create some new zerod parameters to calculate the actual average
	float weightavg = 0.0f;
	float delayavg = 0.0f;
	// Time to test
	int startval = popNums[0]*(popNums[1] - popNums[0]);
	for (int i=0; i < popNums[0]; i++){
		for (int j=popNums[0]; j < popNums[1]; j++){
			// The index is the value of i + (j-initj)*max_i PLUS the initial pop size
			int index = i + (j-popNums[0])*popNums[0] + startval;
			// Check the connections
			REQUIRE(syn.presyns[index] == i);
			REQUIRE(syn.postsyns[index] == j);
			REQUIRE(syn.stdp[index] == 0);
			// Add to the parameters holding an average
			weightavg += syn.weights[index];
			delayavg += (float)syn.delays[index];
		}
	}
	// Calculate the average by dividing
	weightavg = weightavg / (float)startval;
	delayavg = delayavg / (float)startval;
	// Should be within 5% of what the true average is
	REQUIRE(std::abs((weightavg - weightrangeavg)/weightrangeavg) < 0.05);
	REQUIRE(std::abs((delayavg - delayrangeavg)/delayrangeavg) < 0.05);
}
// Next testing what happens when we create a set of Synapses SOLO
TEST_CASE("Synapse ONE_TO_ONE SOLO Creation", "[Synapse]"){
	// Create an instance of Synapse
	Synapse syn;
	// Setting fictional pre and post IDs
	int pre = 2;
	int post = 3;
	// Setting their population numbers
	int popNums[] = {100, 250, 340, 430};
	// Setting up the connection style
	char style[] = "one_to_one";
	// Setting up Weights
	float weightrange[] = {2.0f, 2.0f};
	// Setting up Delays
	int delayrange[] = {2.0f, 2.0f};
	// Determining whether or not there is STDP
	bool stdpswitch = true;
	// Giving the parameter
	float parameter = 0.0f;
	// Creating the Synapses
	syn.AddConnection(pre, 
						post, 
						popNums, 
						style, 
						weightrange,
						delayrange,
						stdpswitch,
						parameter);
	// Time to test
	for (int i=0; i < (popNums[post] - popNums[pre]); i++){
		// The index is the value of i
		int pren = i + popNums[pre-1];
		int postn = i + popNums[post-1];
		// Check the connections
		REQUIRE(syn.presyns[i] == pren);
		REQUIRE(syn.postsyns[i] == postn);
		REQUIRE(syn.weights[i] == 2.0f);
		REQUIRE(syn.delays[i] == 2.0f);
		REQUIRE(syn.stdp[i] == 1);
	}
	// Test the number of connections
	REQUIRE(syn.numconnections == (popNums[post] - popNums[pre]));

	//////////// RANDOM WEIGHTS AND DELAYS ////////////
	// Creating a new set of ranges
	float weightrange_rand[] = {0.1f, 200.0f};
	int delayrange_rand[] = {5, 21};
	// Calculating the specific average of these ranges
	float weightrangeavg = std::abs((weightrange_rand[1] + weightrange_rand[0])/2.0f);
	float delayrangeavg = std::abs((float)(delayrange_rand[1] + delayrange_rand[0])/2.0f);
	// Creating the Synapses
	syn.AddConnection(pre, 
						post, 
						popNums, 
						style, 
						weightrange_rand,
						delayrange_rand,
						stdpswitch,
						parameter);
	// Create some new zerod parameters to calculate the actual average
	float weightavg = 0.0f;
	float delayavg = 0.0f;
	// Time to test
	int startval = std::abs(popNums[post] - popNums[pre]);
	for (int i=0; i < (popNums[post] - popNums[pre]); i++){
		// The index is the value of i + (j-initj)*max_i PLUS the initial pop size
		int index = i + startval;
		// The index is the value of i
		int pren = i + popNums[pre-1];
		int postn = i + popNums[post-1];
		// Check the connections
		REQUIRE(syn.presyns[i] == pren);
		REQUIRE(syn.postsyns[i] == postn);
		REQUIRE(syn.stdp[i] == 1);
		// Add to the parameters holding an average
		weightavg += syn.weights[index];
		delayavg += (float)syn.delays[index];
	}
	// Calculate the average by dividing
	weightavg = weightavg / (float)startval;
	delayavg = delayavg / (float)startval;
	// Should be within 10% of what the true average is (smaller populations)
	REQUIRE(std::abs((weightavg - weightrangeavg)/weightrangeavg) < 0.10);
	REQUIRE(std::abs((delayavg - delayrangeavg)/delayrangeavg) < 0.10);
}
// Next testing what happens when we create random synapses with others
TEST_CASE("Synapse RANDOM + ONE_TO_ONE Creation", "[Synapse]"){
	// Create an instance of Synapse
	Synapse syn;
	// Setting fictional pre and post IDs
	int pre = 2;
	int post = 3;
	// Setting their population numbers
	int popNums[] = {100, 250, 340, 430};
	// Setting up the connection style
	char style[] = "one_to_one";
	// Setting up Weights
	float weightrange[] = {3.0f, 3.0f};
	// Setting up Delays
	int delayrange[] = {1.0f, 1.0f};
	// Determining whether or not there is STDP
	bool stdpswitch = true;
	// Giving the parameter
	float parameter = 0.0f;
	// Creating one_to_one Synapses
	syn.AddConnection(pre, 
						post, 
						popNums, 
						style, 
						weightrange,
						delayrange,
						stdpswitch,
						parameter);
	// Test the number of connections
	REQUIRE(syn.numconnections == (popNums[post] - popNums[pre]));
	// Setting up random connections
	int prer = 1;
	int postr = 0;
	// Random connectivity
	char randstyle[] = "random";
	// Creating a new set of ranges
	float weightrange_rand[] = {0.0f, 600.0f};
	int delayrange_rand[] = {2, 70};
	// Calculating the specific average of these ranges
	float weightrangeavg = std::abs((weightrange_rand[1] + weightrange_rand[0])/2.0f);
	float delayrangeavg = std::abs((float)(delayrange_rand[1] + delayrange_rand[0])/2.0f);
	// stdp off
	bool stdprand = false;
	// set the parameter
	float randparam = 0.5;
	// Create the Random Synapses
	syn.AddConnection(prer, 
						postr, 
						popNums, 
						randstyle, 
						weightrange_rand,
						delayrange_rand,
						stdprand,
						randparam);
	// Time to test the synapses
	// ONE_TO_ONE
	for (int i=0; i < (popNums[post] - popNums[pre]); i++){
		// The index is the value of i
		int pren = i + popNums[pre-1];
		int postn = i + popNums[post-1];
		// Check the connections
		REQUIRE(syn.presyns[i] == pren);
		REQUIRE(syn.postsyns[i] == postn);
		REQUIRE(syn.weights[i] == 3.0f);
		REQUIRE(syn.delays[i] == 1.0f);
		REQUIRE(syn.stdp[i] == 1);
	}
	
	// RANDOM CONNECTIVITIES & WEIGHTS/DELAYS
	int numoneconns = (popNums[post] - popNums[pre]);
	// Check that the number of connections lines up
	int numrand = syn.numconnections - numoneconns;
	float percent = (float) numrand / (float)(popNums[postr]*(popNums[prer]-popNums[prer-1]));
	// Ensure no more than 5% leeway
	REQUIRE(std::abs(percent - randparam) < 0.05);
	// Create some new zerod parameters to calculate the actual average
	float weightavg = 0.0f;
	float delayavg = 0.0f;
	// Check Synapses
	for (int i=0; i < (numrand); i++){
		// Check connections
		REQUIRE(syn.stdp[i+numoneconns] == 0);
		weightavg += syn.weights[i+numoneconns];
		delayavg += syn.delays[i+numoneconns];
	}
	// Calculate the average by dividing
	weightavg = weightavg / (float)numrand;
	delayavg = delayavg / (float)numrand;
	// Should be within 10% of what the true average is (smaller populations)
	REQUIRE(std::abs((weightavg - weightrangeavg)/weightrangeavg) < 0.10f);
	REQUIRE(std::abs((delayavg - delayrangeavg)/delayrangeavg) < 0.10f);
}
// Gaussian Synapse test time.
TEST_CASE("Synapse GAUSSIAN SOLO Creation", "[Synapse]"){
	// Create an instance of Synapse
	Synapse syn;
	// Setting fictional pre and post IDs
	int pre = 0;
	int post = 1;
	// Setting their population numbers
	int popNums[] = {1000, 2000};
	// Setting up the connection style
	char style[] = "gaussian";
	// Setting up Weights
	float weightrange[] = {2.0f, 2.0f};
	// Setting up Delays
	int delayrange[] = {2.0f, 2.0f};
	// Determining whether or not there is STDP
	bool stdpswitch = true;
	// Giving the parameter
	float parameter = 5.0f;
	// Creating the Synapses
	syn.AddConnection(pre, 
						post, 
						popNums, 
						style, 
						weightrange,
						delayrange,
						stdpswitch,
						parameter);
	// The number of synapses is:
	int numgauss = syn.numconnections;
	// Total number of possible synapses is
	int totposs = (std::abs(popNums[post] - popNums[pre]));
	// Initialising the standard deviation check
	float sd = 0;
	// Begin tests
	for (int i=0; i < numgauss; i++){
		// Check the connections
		REQUIRE(syn.weights[i] == 2.0f);
		REQUIRE(syn.delays[i] == 2.0f);
		REQUIRE(syn.stdp[i] == 1);
		// Checking the standard deviation of the connections
		sd += pow(((float)((syn.postsyns[i] - popNums[post-1]) - syn.presyns[i])),2);
	}
	// Checking the standard deviation of the dataset:
	sd = sd/numgauss;
	sd = sqrt(sd);
	// Check that the standard deviation is within 10% of what we expect
	REQUIRE((std::abs((sd-parameter)/parameter)) < 0.1);

	//////////// RANDOM WEIGHTS AND DELAYS ////////////
	// Creating a new set of ranges
	float weightrange_rand[] = {0.1f, 200.0f};
	int delayrange_rand[] = {5, 21};
	// Calculating the specific average of these ranges
	float weightrangeavg = std::abs((weightrange_rand[1] + weightrange_rand[0])/2.0f);
	float delayrangeavg = std::abs((float)(delayrange_rand[1] + delayrange_rand[0])/2.0f);
	// Creating the Synapses
	syn.AddConnection(pre, 
						post, 
						popNums, 
						style, 
						weightrange_rand,
						delayrange_rand,
						stdpswitch,
						parameter);
	// Create some new zerod parameters to calculate the actual average
	float weightavg = 0.0f;
	float delayavg = 0.0f;
	// Time to test
	for (int i=0; i < (syn.numconnections - numgauss); i++){
		// The index is the value of i + (j-initj)*max_i PLUS the initial pop size
		int index = i + numgauss;
		// Check the connections
		REQUIRE(syn.stdp[index] == 1);
		// Add to the parameters holding an average
		weightavg += syn.weights[index];
		delayavg += (float)syn.delays[index];
	}
	// Calculate the average by dividing
	weightavg = weightavg / (float)(syn.numconnections - numgauss);
	delayavg = delayavg / (float)(syn.numconnections - numgauss);
	// Should be within 10% of what the true average is (smaller populations)
	REQUIRE(std::abs((weightavg - weightrangeavg)/weightrangeavg) < 0.10f);
	REQUIRE(std::abs((delayavg - delayrangeavg)/delayrangeavg) < 0.10f);
}










/*

		SPIKE CLASS TESTS

*/
// Unit testing
#include "Spike.h"
TEST_CASE("Spike Constructor", "[Spike]"){
	// Testing the defaults
	Spike sim;
	// Poisson Populations
	REQUIRE(sim.numPoisson == 0);
	REQUIRE(sim.poisson == NULL);
	REQUIRE(sim.poissonrate == NULL);
	REQUIRE(sim.poissonmask == NULL);
	// Things for checking
	REQUIRE(sim.numPops == 0);
	REQUIRE(sim.numConnects == 0);
	// Default parameters
	REQUIRE(sim.timestep == 0.001f);
	REQUIRE(sim.a_plus == 0.005f);
	REQUIRE(sim.a_minus == -0.015f);
	REQUIRE(sim.tau_plus == 0.015f);
	REQUIRE(sim.tau_minus == 0.025f);
	REQUIRE(sim.w_max == 60.0f);
}
// Checking the allocation of a timestep
TEST_CASE("Spike Timestep Setting", "[Spike]"){
	// Creating an instance
	Spike sim;
	// Initially check default timestep
	REQUIRE(sim.timestep == 0.001f);
	// Change timestep and check again
	sim.SetTimestep(0.05f);
	REQUIRE(sim.timestep == 0.05f);
}
// Try creating Poisson Neurons
TEST_CASE("Spike Poisson Neurons", "[Spike]"){
	// Creating an instance
	Spike sim;
	// Parameterise the population
	char poiss[] = "poisson";
	struct neuron_struct rate;
	rate.rate = 30.0f;
	sim.CreateNeurons(1000, poiss, rate);
	// Check the stored variables
	// First, the number of neurons to check
	REQUIRE(sim.population.numNeurons == 1000);
	REQUIRE(sim.population.numperPop[0] == 1000);
	// Check that the values in every part of array
	for (int i = 0; i < sim.population.numNeurons; i++){
		// Set the parameters
		REQUIRE(sim.population.neuronpop_variables[i].parama == 0.0f);
		REQUIRE(sim.population.neuronpop_variables[i].paramb == 0.0f);
		REQUIRE(sim.population.neuronpop_variables[i].paramc == 0.0f);
		REQUIRE(sim.population.neuronpop_variables[i].paramd == 0.0f);
		// Set state variables
		REQUIRE(sim.population.neuronpop_variables[i].state_v == -70.0f);
		REQUIRE(sim.population.neuronpop_variables[i].state_u == 0.0f);
	}
	// Check the poisson specific variables
	REQUIRE(sim.numPoisson == 1);
	REQUIRE(sim.poisson[0][0] == 0);
	REQUIRE(sim.poisson[0][1] == 1000);
	REQUIRE(sim.poissonrate[0] == 30.0f);
}
// Check that the delay created is correct
// Try creating Poisson Neurons
TEST_CASE("Spike Create Connection", "[Spike]"){
	// Creating an instance
	Spike sim;
	// Parameterise the population
	// Types
	char poiss[] = "poisson";
	char izh[] = "izh";
	struct neuron_struct paramCort;
	struct neuron_struct rate;
	paramCort.parama = 0.01f;
	paramCort.paramb = 0.2f;
	paramCort.paramc = -65.0f;
	paramCort.paramd = 8.0f;
	rate.rate = 30.0f;
	// Connectivity
	char one[] = "one_to_one";
	float weights[] = {25.0f, 30.0f};
	float delays[] = {25.0f, 25.0f};
	// Create Populations
	int input = sim.CreateNeurons(1000, poiss, rate);
	int output = sim.CreateNeurons(1000, izh, paramCort);
	// Connect the populations with one-to-one connectivity
	sim.CreateConnection(input, output, one, weights, delays, false, 0.0f);
	// Check the delays
	for (int i=0; i < 1000; i++){
		REQUIRE(sim.synconnects.delays[i] == std::round(25.0f / sim.timestep));
	}
}
// Check the creation of the poisson mask
// Check that the delay created is correct
// Try creating Poisson Neurons
TEST_CASE("Spike Create Poisson Mask", "[Spike]"){
	// Creating an instance
	Spike sim;
	// Parameterise the population
	// Types
	char poiss[] = "poisson";
	char izh[] = "izh";
	struct neuron_struct paramCort;
	struct neuron_struct rate;
	paramCort.parama = 0.01f;
	paramCort.paramb = 0.2f;
	paramCort.paramc = -65.0f;
	paramCort.paramd = 8.0f;
	rate.rate = 30.0f;
	// Create Populations
	int input = sim.CreateNeurons(1000, poiss, rate);
	int output = sim.CreateNeurons(1000, izh, paramCort);
	int lastlayer = sim.CreateNeurons(21, poiss, rate);
	// Check the mask that was created
	for (int i = 0; i < sim.population.numNeurons; i++){
		if ((i < 1000) || (i > 1999)){
			REQUIRE(sim.population.neuronpop_variables[i].rate == rate.rate);
		} else {
			REQUIRE(sim.population.neuronpop_variables[i].rate == 0.0f);
		}
	}
}

// Check the creation of Spike Generator Populations
TEST_CASE("Spike Create Spike Generators", "[Spike]"){
	// Creating an instance
	Spike sim;
	// Parameterise the population
	// Types
	//char poiss[] = "poisson";
	char izh[] = "izh";
	char gen[] = "gen";
	struct neuron_struct paramCort;
	struct neuron_struct rate;
	paramCort.parama = 0.01f;
	paramCort.paramb = 0.2f;
	paramCort.paramc = -65.0f;
	paramCort.paramd = 8.0f;
	rate.rate = 0.0f;
	// Generator stuff
	// Stimulus 1
	int gIDs[] = {0, 1, 3, 500, 999};
	float gtimes[] = {0.07f, 0.09f, 0.18f, 0.25f, 2.00f};
	// Stimulus 2
	int gIDs_two[] = {5, 500, 6, 2, 10, 5};
	float gtimes_two[] = {0.7f, 1.09f, 0.18f, 0.25f, 2.01f, 0.06f};
	// Create Populations
	// int input = sim.CreateNeurons(1000, poiss, rate);
	int output = sim.CreateNeurons(1000, izh, paramCort);
	// Creating the Neurons that I wish to have in my generator population
	int lastlayer = sim.CreateNeurons(1000, gen, rate);
	// Providing this layer a set of firing times
	sim.CreateGenerator(lastlayer, 0, 5, gIDs, gtimes);
	sim.CreateGenerator(lastlayer, 1, 6, gIDs_two, gtimes_two);
	// Check the generator population is correctly created
	REQUIRE(sim.numStimuli == 2);
	REQUIRE(sim.numEntries[0] == 5);
	REQUIRE(sim.numEntries[1] == 6);
	
	// Check that the correct spikes exist
	for (int i = 0; i < sim.numEntries[0]; i++){
		REQUIRE(sim.genids[0][i] == (gIDs[i] + 1000));
		REQUIRE(sim.gentimes[0][i] == gtimes[i]);
	}
	for (int i = 0; i < sim.numEntries[1]; i++){
		REQUIRE(sim.genids[1][i] == (gIDs_two[i] + 1000));
		REQUIRE(sim.gentimes[1][i] == gtimes_two[i]);
	}
	// Check total number of neurons
	REQUIRE(sim.population.numNeurons == 2000);
	// Run this network
	sim.Run(2.5f, 1, true);
	// Open the output file and check!
	ifstream spikeidfile, spiketimesfile;
	spikeidfile.open("results/Epoch0_SpikeIDs.bin", ios::in | ios::binary | ios::ate);
	spiketimesfile.open("results/Epoch0_SpikeTimes.bin", ios::in | ios::binary | ios::ate);

	// A variable to hold the size of my file
	streampos sizeids, sizetimes;
	// Calculate the number of bytes long that this file is
	sizeids = spikeidfile.tellg();
	sizetimes = spiketimesfile.tellg();
	REQUIRE(sizeids == sizetimes);
	// Create variables to hold the results
	int spikeids[int(sizeids/4)];
	float spiketimes[int(sizetimes/4)];
	// Begin at the starts of the files
	spikeidfile.seekg(0, ios::beg);
	spiketimesfile.seekg(0, ios::beg);
	// Read all of the bytes
	spikeidfile.read((char*)&spikeids, sizeids);
	spiketimesfile.read((char*)&spiketimes, sizetimes);
	// Close the files
	spikeidfile.close();
	spiketimesfile.close();
	for (int i = 0; i < (sim.numEntries[0] + sim.numEntries[1]); i++){
		printf("%d\n", spikeids[i]);
	}
	// Test whether the output is correct!
	for (int i = 0; i < sim.numEntries[0]; i++){
		REQUIRE(spikeids[i] == (gIDs[i] + 1000));
		REQUIRE(spiketimes[i] == gtimes[i]);
	}
	// I must order the second list to check
	int gIDs_two_test[] = {5, 6, 2, 5, 500, 10};
	float gtimes_two_test[] = {0.06f, 0.18f, 0.25f, 0.7f, 1.09f, 2.01f};
	for (int i = sim.numEntries[0]; i < (sim.numEntries[0] + sim.numEntries[1]); i++){
		REQUIRE(spikeids[i] == (gIDs_two_test[i - sim.numEntries[0]] + 1000));
		REQUIRE((spiketimes[i] - gtimes_two_test[i - sim.numEntries[0]]) < (sim.timestep*0.2));
	}
}













#ifdef GPU_TEST
/*

		GPU CODE TESTS

*/
//	CUDA library
#include <cuda.h>
// Unit testing
#include "CUDAcode.h"
// I must ensure that I carry out the correct error checking:

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

// The two functions that we can use
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

// When we wish to check for errors in functions such as malloc directly
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
// If we want to do error-checking
#ifdef CUDA_ERROR_CHECK
	// Check for success
    if ( cudaSuccess != err )
    {
    	// Output the issue if there is one
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// When we wish to check that functions did not introduce errors
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	// Get last error (i.e. in the function that has just run)
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

TEST_CASE("CUDA Poisson Update Test", "[Spike]"){
	// Get the default timestep
	// Create instance of Spike
	Spike sim;
	// Creating some example data
	size_t numNeurons = 10;
	float h_randoms[] = {0.0001f, 0.9f, 0.08f, 0.007f, 0.006f, 0.5f, 0.04f, 0.003f, 0.0002f, 0.1f};
	float h_neuron_v[] = {-30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f};
	float h_neuron_u[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
	float h_poisson_rate[] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 200.0f};
	struct neuron_struct h_neuronpop_vars[10];
	for (int i = 0; i < 10; i++){
		h_neuronpop_vars[i].state_v = h_neuron_v[i];
		h_neuronpop_vars[i].state_u = h_neuron_u[i];
		h_neuronpop_vars[i].rate = h_poisson_rate[i];
	}
	// Setting space on the device
	float* d_randoms;
	struct neuron_struct *d_neuronpop_vars;
	CudaSafeCall(cudaMalloc((void **)&d_randoms, sizeof(float)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_randoms, h_randoms, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_neuronpop_vars, h_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyHostToDevice));
	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
	poisupdate<<<blocks, threads>>>( d_randoms, 
								d_neuronpop_vars,
								sim.timestep,
								numNeurons);
	CudaCheckError();
	// Copying the data back to the host
	CudaSafeCall(cudaMemcpy(h_neuronpop_vars, d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyDeviceToHost));
	// Free device memory
	CudaSafeCall(cudaFree(d_randoms));
	CudaSafeCall(cudaFree(d_neuronpop_vars));

	// Run the tests!
	for (int i = 0; i < numNeurons; i++){
		if (h_neuronpop_vars[i].rate != 0.0f){
			if (h_randoms[i] < h_neuronpop_vars[i].rate*sim.timestep){
				REQUIRE(h_neuronpop_vars[i].state_v == 35.0f);
				REQUIRE(h_neuronpop_vars[i].state_u == 0.0f);
			} else {
				REQUIRE(h_neuronpop_vars[i].state_v == -70.0f);
				REQUIRE(h_neuronpop_vars[i].state_u == 0.0f);
			}

		} else {
			REQUIRE(h_neuronpop_vars[i].state_v == -30.0f);
			REQUIRE(h_neuronpop_vars[i].state_u == 0.5f);
		}
	}
}

TEST_CASE("CUDA Current Calculation Test", "[Spike]"){
	// Creating some example data
	int h_spikes[] = {0, 1, 2, 3, 4, 2, 3, 1, 2};
	int h_postsyns[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
	float h_weights[] = {-30.0f, 10.0f, 5.0f, 7.0f, 11.0f, 13.6f, 40.1f, 76.0f, -3.0f};
	float h_lastactive[] = {0.5f, 1.0f, 7.5f, 6.3f, -100.0f, 7.11f, 9.5f, 1.5f, 0.1f};
	float h_currinj[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	float currtime = 3.789f;
	size_t numConns = 9;
	size_t numNeurons = 10;

	// Setting space on the device
	int* d_spikes;
	int* d_postsyns;
	float* d_weights;
	float* d_lastactive;
	float* d_currinj;
	

	CudaSafeCall(cudaMalloc((void **)&d_spikes, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_postsyns, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_weights, sizeof(float)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_lastactive, sizeof(float)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_currinj, sizeof(float)*numNeurons));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_spikes, h_spikes, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_postsyns, h_postsyns, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_weights, h_weights, sizeof(float)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_lastactive, h_lastactive, sizeof(float)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_currinj, h_currinj, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));
	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
	currentcalc<<<blocks, threads>>>( d_spikes, 
								d_weights, 
								d_lastactive,  
								d_postsyns,
								d_currinj,
								currtime,
								numConns,
								numNeurons);
	CudaCheckError();
	// Copying the data back to the host
	int e_spikes[] = {0, 1, 2, 3, 4, 2, 3, 1, 2, 5};
	float e_lastactive[] = {0.5f, 1.0f, 7.5f, 6.3f, -100.0f, 7.11f, 9.5f, 1.5f, 0.1f, 0.9f};
	float e_currinj[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	CudaSafeCall(cudaMemcpy(e_spikes, d_spikes, sizeof(int)*numConns, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(e_lastactive, d_lastactive, sizeof(float)*numConns, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(e_currinj, d_currinj, sizeof(float)*numNeurons, cudaMemcpyDeviceToHost));

	// Run the tests!
	for (int i = 0; i < numConns; i++){
		// Require that spike countdown decremented
		REQUIRE(e_spikes[i] == (h_spikes[i] - 1));
		// If it is at zero
		if (e_spikes[i] == 0) {
			// That post synapse's last active time should be now
			REQUIRE(e_lastactive[i] == currtime);
			// The location of the post synaptic neuron should have an incoming current
			REQUIRE(e_currinj[h_postsyns[i]] == h_weights[i]);
		}
	}


	// Free device memory
	CudaSafeCall(cudaFree(d_spikes));
	CudaSafeCall(cudaFree(d_postsyns));
	CudaSafeCall(cudaFree(d_weights));
	CudaSafeCall(cudaFree(d_lastactive));
	CudaSafeCall(cudaFree(d_currinj));
}

TEST_CASE("CUDA LTD Test", "[Spike]"){
	// Creating some example data
	int h_stdp[] = {0, 0, 0, 0, 1, 1, 1, 1, 1};
	int h_postsyns[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
	float h_weights[] = {-30.0f, 10.0f, 5.0f, 7.0f, 11.0f, 13.6f, 40.1f, 76.0f, -3.0f};
	float h_lastactive[] = {3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f};
	float h_lastspiketime[] = {3.780f, 3.779f, 2.789f, 2.589f, 1.986f, 3.789f, 3.700f, 3.182f, 2.854f, 1.678f};
	float currtime = 3.790f;
	float w_max = 60.0f;
	float a_minus = -0.015f;
	float tau_minus = 0.025f;
	size_t numConns = 9;
	size_t numNeurons = 10;

	// Setting space on the device
	int* d_stdp;
	int* d_postsyns;
	float* d_weights;
	float* d_lastactive;
	float* d_lastspiketime;	
	CudaSafeCall(cudaMalloc((void **)&d_stdp, sizeof(int)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_postsyns, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_weights, sizeof(float)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_lastactive, sizeof(float)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_lastspiketime, sizeof(float)*numNeurons));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_stdp, h_stdp, sizeof(int)*numNeurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_postsyns, h_postsyns, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_weights, h_weights, sizeof(float)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_lastactive, h_lastactive, sizeof(float)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_lastspiketime, h_lastspiketime, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));
	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
	ltdweights<<<blocks, threads>>>(d_lastactive,
							d_weights,
							d_stdp,
							d_lastspiketime,
							d_postsyns,
							currtime,
							w_max,
							a_minus,
							tau_minus,
							numConns,
							numNeurons);
	CudaCheckError();
	// Copying the data back to the host
	float e_weights[] = {-30.0f, 10.0f, 5.0f, 7.0f, 11.0f, 13.6f, 40.1f, 76.0f, -3.0f};
	CudaSafeCall(cudaMemcpy(e_weights, d_weights, sizeof(float)*numConns, cudaMemcpyDeviceToHost));

	// Run the tests!
	for (int i = 0; i < numConns; i++){
		// Check that LTD has been correctly carried out:
		if (h_stdp[i] == 1){
			float change = w_max*(a_minus*exp((h_lastspiketime[h_postsyns[i]] - h_lastactive[i])/tau_minus));
			REQUIRE(e_weights[i] == (h_weights[i] + change));
		} else {
			REQUIRE(e_weights[i] == h_weights[i]);
		}
	}


	// Free device memory
	CudaSafeCall(cudaFree(d_stdp));
	CudaSafeCall(cudaFree(d_postsyns));
	CudaSafeCall(cudaFree(d_weights));
	CudaSafeCall(cudaFree(d_lastactive));
	CudaSafeCall(cudaFree(d_lastspiketime));
}

TEST_CASE("CUDA State Update Test", "[Spike]"){
	// Get the default timestep
	// Create instance of Spike
	Spike sim;
	// Creating some example data
	float h_neuron_v[] = {-30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f};
	float h_neuron_u[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
	float h_currinj[] = {0.0f, 1.0f, 2.0f, 0.3f, 40.0f, 0.005f, 0.12f, 0.67f, 0.0f, 1.0f};
	float h_parama[] = {0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f};
	float h_paramb[] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, -0.1f, -0.1f, -0.1f, -0.1f, -0.1f};
	size_t numNeurons = 10;
	struct neuron_struct h_neuronpop_vars[10];
	for (int i = 0; i < 10; i++){
		h_neuronpop_vars[i].state_v = h_neuron_v[i];
		h_neuronpop_vars[i].state_u = h_neuron_u[i];
		h_neuronpop_vars[i].parama = h_parama[i];
		h_neuronpop_vars[i].paramb = h_paramb[i];
	}

	// Setting space on the device
	struct neuron_struct *d_neuronpop_vars;
	float* d_currinj;	
	CudaSafeCall(cudaMalloc((void **)&d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_currinj, sizeof(float)*numNeurons));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_neuronpop_vars, h_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_currinj, h_currinj, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));

	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
	stateupdate<<<blocks, threads>>>(d_neuronpop_vars,
							d_currinj,
							sim.timestep,
							numNeurons);
	CudaCheckError();
	// Copying the data back to the host
	struct neuron_struct e_neuronpop_vars[10];
	CudaSafeCall(cudaMemcpy(e_neuronpop_vars, d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyDeviceToHost));

	// Run the tests!
	for (int i = 0; i < numNeurons; i++){
		// Check that states have been updated correctly
		float delta_v = 0.04f*h_neuronpop_vars[i].state_v*h_neuronpop_vars[i].state_v + 5.0f*h_neuronpop_vars[i].state_v + 140 - h_neuronpop_vars[i].state_u;
		REQUIRE(std::abs(e_neuronpop_vars[i].state_v - (h_neuronpop_vars[i].state_v + (sim.timestep*1000.0f)*delta_v + h_currinj[i])) < 0.0001);
		float new_v = (h_neuronpop_vars[i].state_v + (sim.timestep*1000.0f)*delta_v + h_currinj[i]);
		float delta_u = h_neuronpop_vars[i].parama*(h_neuronpop_vars[i].paramb*new_v - h_neuronpop_vars[i].state_u);
		REQUIRE(std::abs(e_neuronpop_vars[i].state_u - (h_neuronpop_vars[i].state_u + (sim.timestep*1000.0f)*delta_u)) < 0.0001);
	}


	// Free device memory
	CudaSafeCall(cudaFree(d_neuronpop_vars));
	CudaSafeCall(cudaFree(d_currinj));
}

TEST_CASE("CUDA Spiking Neurons Test", "[Spike]"){
	// Creating some example data
	float h_neuron_v[] = {-30.0f, 30.0f, -20.0f, 20.0f, -40.0f, 40.0f, -50.0f, 50.0f, 90.0f, -3.0f};
	float h_neuron_u[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
	float h_lastspiketime[] = {3.780f, 3.779f, 2.789f, 2.589f, 1.986f, 3.789f, 3.700f, 3.182f, 2.854f, 1.678f};
	float h_paramc[] = {-65.0f, -65.0f, -65.0f, -65.0f, -65.0f, -55.0f, -55.0f, -55.0f, -55.0f, -55.0f};
	float h_paramd[] = {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f};
	float currtime = 3.790f;
	size_t numNeurons = 10;
	struct neuron_struct h_neuronpop_vars[10];
	for (int i = 0; i < 10; i++){
		h_neuronpop_vars[i].state_v = h_neuron_v[i];
		h_neuronpop_vars[i].state_u = h_neuron_u[i];
		h_neuronpop_vars[i].paramc = h_paramc[i];
		h_neuronpop_vars[i].paramd = h_paramd[i];
	}

	// Setting space on the device
	float* d_lastspiketime;
	struct neuron_struct * d_neuronpop_vars;
	CudaSafeCall(cudaMalloc((void **)&d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_lastspiketime, sizeof(float)*numNeurons));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_neuronpop_vars, h_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_lastspiketime, h_lastspiketime, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));
	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
	spikingneurons<<<blocks, threads>>>(d_neuronpop_vars,
								d_lastspiketime,
								currtime,
								numNeurons);
	CudaCheckError();
	// Copying the data back to the host
	float e_lastspiketime[] = {3.780f, 3.779f, 2.789f, 2.589f, 1.986f, 3.789f, 3.700f, 3.182f, 2.854f, 1.678f};
	struct neuron_struct e_neuronpop_vars[10];
	CudaSafeCall(cudaMemcpy(e_lastspiketime, d_lastspiketime, sizeof(float)*numNeurons, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(e_neuronpop_vars, d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyDeviceToHost));
	// Run the tests!
	for (int i = 0; i < numNeurons; i++){
		// Check for spiking
		if (h_neuron_v[i] >= 30.0f) {
			// Check that the neuron states have been corrected
			REQUIRE(e_neuronpop_vars[i].state_v == h_neuronpop_vars[i].paramc);
			REQUIRE(e_neuronpop_vars[i].state_u == (h_neuronpop_vars[i].state_u + h_neuronpop_vars[i].paramd));
			// Reapply the spike times
			REQUIRE(e_lastspiketime[i] == currtime);
		} else 	{
			// If not, ensure that things are still right
			REQUIRE(e_neuronpop_vars[i].state_v == h_neuronpop_vars[i].state_v);
			REQUIRE(e_neuronpop_vars[i].state_u == h_neuronpop_vars[i].state_u);
			REQUIRE(e_lastspiketime[i] == h_lastspiketime[i]);
		}
	}
	// Free device memory
	CudaSafeCall(cudaFree(d_neuronpop_vars));
	CudaSafeCall(cudaFree(d_lastspiketime));
}

TEST_CASE("CUDA Spiking Synapse Test", "[Spike]"){
	// Creating some example data
	int h_presyns[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
	int h_spikes[] = {0, -1, 2, 0, 4, -1, 3, -1, -22};
	int h_spikebuffer[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
	int h_delays[] = {50, 12, 2, 43, 4, 7, 3, 11, 2};
	float h_lastspiketime[] = {3.790f, 3.779f, 3.790f, 2.589f, 1.986f, 3.790f, 3.700f, 3.790f, 2.854f, 3.790f};
	float currtime = 3.790f;
	size_t numConns = 9;
	size_t numNeurons = 10;

	// Setting space on the device
	int* d_presyns;
	int* d_spikes;
	int* d_spikebuffer;
	int* d_delays;
	float* d_lastspiketime;

	CudaSafeCall(cudaMalloc((void **)&d_presyns, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_spikes, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_spikebuffer, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_delays, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_lastspiketime, sizeof(float)*numNeurons));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_presyns, h_presyns, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_spikes, h_spikes, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_spikebuffer, h_spikebuffer, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_delays, h_delays, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_lastspiketime, h_lastspiketime, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));

	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
	synapsespikes<<<blocks, threads>>>(d_presyns,
								d_delays,
								d_spikes,
								d_lastspiketime,
								d_spikebuffer,
								currtime,
								numConns,
								numNeurons);
	CudaCheckError();
	// Copying the data back to the host
	int e_spikes[] = {0, -1, 2, 0, 4, -1, 3, -1, -22};
	CudaSafeCall(cudaMemcpy(e_spikes, d_spikes, sizeof(int)*numConns, cudaMemcpyDeviceToHost));

	// Run the tests!
	for (int i =0; i < numConns; i++){
		// Check if this synapse's neuron fired
		if (h_lastspiketime[h_presyns[i]] == currtime){
			// Check if it should be updated
			if (h_spikes[i] <= 0) {
				// Update according to the delay
				REQUIRE(e_spikes[i] == h_delays[i]);
			} else {
				// Or it should remain the same
				REQUIRE(e_spikes[i] == h_spikes[i]);
			}
		// If it has gotten too negative, it should be reset to -1
		} else if (h_spikes[i] < 0) {
			REQUIRE(e_spikes[i] == -1);
		}
	}
	// Free device memory
	CudaSafeCall(cudaFree(d_presyns));
	CudaSafeCall(cudaFree(d_spikes));
	CudaSafeCall(cudaFree(d_lastspiketime));
	CudaSafeCall(cudaFree(d_delays));
}

TEST_CASE("CUDA LTP Test", "[Spike]"){
	// Creating some example data
	int h_stdp[] = {0, 0, 0, 0, 1, 1, 1, 1, 1};
	int h_postsyns[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
	float h_weights[] = {-30.0f, 10.0f, 5.0f, 7.0f, 11.0f, 13.6f, 40.1f, 76.0f, -3.0f};
	float h_lastspiketime[] = {3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790f, 3.790};
	float h_lastactive[] = {3.780f, 3.779f, 2.789f, 2.589f, 1.986f, 3.789f, 3.700f, 3.182f, 2.854f};
	float w_max = 60.0f;
	float a_plus = 0.005f;
	float tau_plus = 0.015f;
	float currtime = 3.790f;
	size_t numConns = 9;
	size_t numNeurons = 10;

	// Setting space on the device
	int* d_stdp;
	int* d_postsyns;
	float* d_weights;
	float* d_lastactive;
	float* d_lastspiketime;	
	CudaSafeCall(cudaMalloc((void **)&d_stdp, sizeof(int)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_postsyns, sizeof(int)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_weights, sizeof(float)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_lastactive, sizeof(float)*numConns));
	CudaSafeCall(cudaMalloc((void **)&d_lastspiketime, sizeof(float)*numNeurons));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_stdp, h_stdp, sizeof(int)*numNeurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_postsyns, h_postsyns, sizeof(int)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_weights, h_weights, sizeof(float)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_lastactive, h_lastactive, sizeof(float)*numConns, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_lastspiketime, h_lastspiketime, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));
	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
	synapseLTP<<<blocks, threads>>>(d_postsyns,
							d_lastspiketime,
							d_stdp,
							d_lastactive,
							d_weights,
							a_plus,
							tau_plus,
							w_max,
							currtime,
							numConns,
							numNeurons);
	CudaCheckError();
	// Copying the data back to the host
	float e_weights[] = {-30.0f, 10.0f, 5.0f, 7.0f, 11.0f, 13.6f, 40.1f, 76.0f, -3.0f};
	CudaSafeCall(cudaMemcpy(e_weights, d_weights, sizeof(float)*numConns, cudaMemcpyDeviceToHost));

	// Run the tests!
	for (int i = 0; i < numConns; i++){
		// Check that LTP has been correctly carried out:
		if ((h_stdp[i] == 1) && (h_lastspiketime[h_postsyns[i]] == currtime)){
			float change = (w_max - h_weights[i])*(a_plus*expf(-(currtime - h_lastactive[i])/tau_plus));
			REQUIRE(e_weights[i] == (h_weights[i] + change));
		} else {
			REQUIRE(e_weights[i] == h_weights[i]);
		}
	}


	// Free device memory
	CudaSafeCall(cudaFree(d_stdp));
	CudaSafeCall(cudaFree(d_postsyns));
	CudaSafeCall(cudaFree(d_weights));
	CudaSafeCall(cudaFree(d_lastactive));
	CudaSafeCall(cudaFree(d_lastspiketime));
}

TEST_CASE("CUDA Spike Generator Update Test", "[Spike]"){
	// Get the default timestep
	// Create instance of Spike
	Spike sim;
	// Creating some example data
	size_t numNeurons = 10;
	float h_neuron_v[] = {-30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f, -30.0f};
	float h_neuron_u[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
	int h_genids[] = {0, 2, 4, 6, 8};
	float h_gentimes[] = {0.0f, 3.789f, 3.789f, 0.01f, 3.789f};
	float currtime = 3.789f;
	size_t numEntries = 5;
	// Setting space on the device
	struct neuron_struct h_neuronpop_vars[10];
	for (int i = 0; i < 10; i++){
		h_neuronpop_vars[i].state_v = h_neuron_v[i];
		h_neuronpop_vars[i].state_u = h_neuron_u[i];
	}
	int* d_genids;
	float* d_gentimes;
	struct neuron_struct *d_neuronpop_vars;
	CudaSafeCall(cudaMalloc((void **)&d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_genids, sizeof(int)*numEntries));
	CudaSafeCall(cudaMalloc((void **)&d_gentimes, sizeof(float)*numEntries));
	// Copying the data to the device
	CudaSafeCall(cudaMemcpy(d_neuronpop_vars, h_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_gentimes, h_gentimes, sizeof(float)*numEntries, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_genids, h_genids, sizeof(int)*numEntries, cudaMemcpyHostToDevice));
	// Setting up dimensions
	dim3 threads(100,1,1);
	dim3 blocks(1,1,1);
	// Running the function
    genupdate<<<blocks, threads>>> (d_neuronpop_vars,
									d_genids,
									d_gentimes,
									currtime,
									sim.timestep,
									numEntries);
	CudaCheckError();
	// Copying the data back to the host
	struct neuron_struct e_neuronpop_vars[10];
	CudaSafeCall(cudaMemcpy(e_neuronpop_vars, d_neuronpop_vars, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyDeviceToHost));

	// Run the tests!
	for (int i = 0; i < numEntries; i++){
		if (h_gentimes[i] == currtime) {
			REQUIRE(e_neuronpop_vars[h_genids[i]].state_v == 35.0f);
			REQUIRE(e_neuronpop_vars[h_genids[i]].state_u == 0.0f);
		} else {
			REQUIRE(e_neuronpop_vars[h_genids[i]].state_v == -70.0f);
			REQUIRE(e_neuronpop_vars[h_genids[i]].state_u == 0.0f);
		}
	}
	// Free device memory
	CudaSafeCall(cudaFree(d_neuronpop_vars));
	CudaSafeCall(cudaFree(d_genids));
	CudaSafeCall(cudaFree(d_gentimes));
}

#endif






