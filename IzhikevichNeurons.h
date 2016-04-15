#ifndef IzhikevichNeurons_H
#define IzhikevichNeurons_H

//	CUDA library
#include <cuda.h>

#include "Structs.h"

#include "Neurons.h"

struct izhikevich_neuron_struct : neuron_struct {
	izhikevich_neuron_struct(): test(0.0f) { neuron_struct(); }   // default Constructor

	float test;
};

class IzhikevichNeurons : public Neurons {
public:
	// Constructor/Destructor
	IzhikevichNeurons();
	~IzhikevichNeurons();

	izhikevich_neuron_struct * d_izhikevich_neuron_variables;
	izhikevich_neuron_struct * izhikevich_neuron_variables;


	virtual int AddGroupNew(neuron_struct *params, int shape[2]);


	virtual void initialise_device_pointersNew();
	virtual void reset_neuron_variables_and_spikesNew();
	// void set_threads_per_block_and_blocks_per_grid(int threads);


	// void poisupdate_wrapper(float* d_randoms, float timestep);

	// void genupdate_wrapper(int* genids,
	// 						float* gentimes,
	// 						float currtime,
	// 						float timestep,
	// 						size_t numEntries,
	// 						int genblocknum, 
	// 						dim3 threadsPerBlock);

	// void spikingneurons_wrapper(float currtime);

	// void stateupdate_wrapper(float* current_injection,
	// 						float timestep);



};




#endif