if (BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Please do not use globbing: list targets explicitly!
# Listing targets / files makes for much easier debugging
# and clearer collaboration!
#
# See http://stackoverflow.com/a/18538444 for more info.

add_subdirectory(Backend)
add_subdirectory(Helpers)
add_subdirectory(Models)
add_subdirectory(Neurons)
add_subdirectory(Optimiser)
add_subdirectory(Plotting)
add_subdirectory(RecordingElectrodes)
add_subdirectory(Simulator)
add_subdirectory(SpikeAnalyser)
add_subdirectory(STDP)
add_subdirectory(Synapses)

add_library(Spike
  $<TARGET_OBJECTS:Backend>
  $<TARGET_OBJECTS:Helpers>
  $<TARGET_OBJECTS:Models>
  $<TARGET_OBJECTS:Neurons>
  $<TARGET_OBJECTS:Optimiser>
  $<TARGET_OBJECTS:RecordingElectrodes>
  $<TARGET_OBJECTS:Simulator>
  $<TARGET_OBJECTS:SpikeAnalyser>
  $<TARGET_OBJECTS:STDP>
  $<TARGET_OBJECTS:Synapses>
  )

# target_compile_features(Spike PUBLIC cxx_nullptr)

target_link_libraries(Spike PRIVATE -Wl,--whole-archive
  SpikeDummyHelpers 
  SpikeDummyNeurons
  SpikeDummyRecordingElectrodes
  SpikeDummySpikeAnalyser
  SpikeDummySTDP
  SpikeDummySynapses
  -Wl,--no-whole-archive)

if (BUILD_WITH_CUDA)
  target_link_libraries(Spike PRIVATE -Wl,--whole-archive
    SpikeCUDAHelpers
    SpikeCUDANeurons
    SpikeCUDARecordingElectrodes
    SpikeCUDASpikeAnalyser
    SpikeCUDASTDP
    SpikeCUDASynapses
    -Wl,--no-whole-archive)
endif()

if (ENABLE_PLOTTING)
  target_link_libraries(Spike SpikePlotting)
endif()
