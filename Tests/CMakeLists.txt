# add_library(Catch INTERFACE)
# target_include_directories(Catch INTERFACE ${YOUR_CATCH_INCLUDE_DIR})

foreach(test
    Test_Helpers
#    Test_Neurons
    Test_RecordingElectrode
#    Test_STDP
    Test_Simulator
    Test_SpikeAnalyser
#    Test_Synapses
    )
  add_executable(${test} ${test}.cpp TestMain.cpp)
  target_link_libraries(${test} Spike) # Catch)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
