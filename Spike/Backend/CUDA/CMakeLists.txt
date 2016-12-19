if(BUILD_SHARED_LIBS)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler -fPIC")
endif()

add_subdirectory(Helpers)
add_subdirectory(Neurons)
add_subdirectory(RecordingElectrodes)
add_subdirectory(SpikeAnalyser)
add_subdirectory(STDP)
add_subdirectory(Synapses)

