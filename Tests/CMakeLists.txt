include(CTest)

# add_library(Catch INTERFACE)
# target_include_directories(Catch INTERFACE ${YOUR_CATCH_INCLUDE_DIR})

enable_testing()

foreach(test
    Test_Helpers)
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} Catch Spike)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
