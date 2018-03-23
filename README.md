# Spike
http://oftnai.github.io/Spike/

A GPGPU based Spiking Neural Network (SNN) designed to provide users flexibility in the creation of simulations and high speed in execution. Written in C++/CUDA. (Distributed under the MIT License, See LICENSE)

# Requirements
  - Windows 8 (or greater), OS X 10.10.5 (or greater), (Soon to be tested on Linux)
  - NVIDIA GPU capable of CUDA code execution
  - NVIDIA CUDA Toolkit >= v7.5
  - Catch.hpp v1.5.6 (https://github.com/philsquared/Catch/tree/master/single_include)

Other packages:
  - MathGL (http://sourceforge.net/projects/mathgl/files/): Recommended, allows the creation of plots
  - Dakota (https://dakota.sandia.gov/content/packages): Recommended for parameter optimization using genetic algorithms

For installation instructions, see "Install.txt" file.

# Release Log

v1.0 (13/07/2016):  
  - Overhaul of code structure for fully separated Neuron, Synapse, STDP and Simulator Classes.
  - Addition of LIF Neuron type and Conductance based Synapses
  - Ability to create 2D Neuron Layers and Synapse Connectivities
  - Scripts for Gabor filtering of images (to represent V1 simple cell outputs)
  - "Tests" folder containing unittests. Created using Catch (https://github.com/philsquared/Catch) v1.5.6
  - Plotting functionality reliant upon the MathGL library (mathgl.sourceforge.net/)
  - Install.txt file descriping the steps necessary for installation
  - Example Networks in the "Experiments" folder


This simulator is under constant development. We recommend users to regularly update any codebase which they use. This software is regularly tested on Mac OSX (El Capitan) and Windows 8 using NVIDIA GPUs (e.g. Geforce GTX 980, Geforce GTX 980ti, Geforce GTX 1080 etc.) The software has been tested on CUDA Toolkit 7.5 and above.

Any individuals who have any questions or wish to contribute, please contact: 

Co-Creators: Nasir Ahmad (nasir.ahmad@psy.ox.ac.uk) & James Isbister (james.isbister@psy.ox.ac.uk)

Contributors: Akihiro Eguchi (akihiro.eguchi@psy.ox.ac.uk)
