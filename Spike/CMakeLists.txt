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
add_subdirectory(Plasticity)
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
  $<TARGET_OBJECTS:Plasticity>
  $<TARGET_OBJECTS:Synapses>
  )

# target_compile_features(Spike PUBLIC cxx_nullptr)

set(WHOLE_ARCHIVE_FLAG "-Wl,--whole-archive")
set(NO_WHOLE_ARCHIVE_FLAG "-Wl,--no-whole-archive")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(WHOLE_ARCHIVE_FLAG "-Wl,-all_load")
  set(NO_WHOLE_ARCHIVE_FLAG "-Wl,-noall_load")
endif()

target_link_libraries(Spike PRIVATE ${WHOLE_ARCHIVE_FLAG}
  SpikeDummyHelpers 
  SpikeDummyNeurons
  SpikeDummyRecordingElectrodes
  SpikeDummySpikeAnalyser
  SpikeDummyPlasticity
  SpikeDummySynapses
  ${NO_WHOLE_ARCHIVE_FLAG})

if (BUILD_WITH_CUDA)
  target_link_libraries(Spike PRIVATE ${WHOLE_ARCHIVE_FLAG}
    SpikeCUDAHelpers
    SpikeCUDANeurons
    SpikeCUDARecordingElectrodes
    SpikeCUDASpikeAnalyser
    SpikeCUDAPlasticity
    SpikeCUDASynapses
    ${NO_WHOLE_ARCHIVE_FLAG})
endif()

if (ENABLE_PLOTTING)
  target_link_libraries(Spike SpikePlotting)
endif()
