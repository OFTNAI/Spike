#ifndef InformationAnalysis_H
#define InformationAnalysis_H

#include <cuda.h>

#include "Simulator.h"

class InformationAnalysis{
public:

	// Constructor/Destructor
	InformationAnalysis(Simulator * simulator_parameter);
	~InformationAnalysis();

	Simulator * simulator;

};

#endif