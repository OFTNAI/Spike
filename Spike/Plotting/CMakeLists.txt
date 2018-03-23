if (ENABLE_PLOTTING)
  add_library(SpikePlotting STATIC
    Plotter.cpp
    )
  target_compile_options(SpikePlotting PRIVATE "-fext-numeric-literals")
endif()
