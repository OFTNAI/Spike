#ifndef SPIKE_H
#define SPIKE_H

// Models and Simulators
#include "Spike/Models/SpikingModel.hpp"

// Neuron Models 
#include "Spike/Neurons/LIFSpikingNeurons.hpp"

// Input Neuron Models 
#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
#include "Spike/Neurons/PatternedPoissonInputSpikingNeurons.hpp"
#include "Spike/Neurons/ImagePoissonInputSpikingNeurons.hpp"
#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"

// Plasticity Rules 
#include "Spike/Plasticity/EvansSTDPPlasticity.hpp"
#include "Spike/Plasticity/VogelsSTDPPlasticity.hpp"
#include "Spike/Plasticity/WeightDependentSTDPPlasticity.hpp"
#include "Spike/Plasticity/WeightNormSTDPPlasticity.hpp"

// Synapses
#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
#include "Spike/Synapses/VoltageSpikingSynapses.hpp"

// Monitors
#include "Spike/ActivityMonitor/SpikingActivityMonitor.hpp"
#include "Spike/ActivityMonitor/RateActivityMonitor.hpp"

#endif
