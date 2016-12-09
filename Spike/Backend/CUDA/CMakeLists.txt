CUDA_ADD_LIBRARY(SpikeCUDAMisc
  RandomStateManager.cu
  STATIC)

add_subdirectory(Neurons)
add_subdirectory(RecordingElectrodes)
add_subdirectory(SpikeAnalyser)
add_subdirectory(STDP)
add_subdirectory(Synapses)

