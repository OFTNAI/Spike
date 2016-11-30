CUDA_ADD_LIBRARY(SpikeCUDASynapses
  Synapses.cu
  SpikingSynapses.cu
  ConductanceSpikingSynapses.cu
  CurrentSpikingSynapses.cu
  STATIC)
