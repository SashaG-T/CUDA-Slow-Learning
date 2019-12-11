# CUDA-Slow-Learning
A CUDA application that simulates a small neural network of random input and random weights.

The application chooses random weights to reward when certain neurons are activated.
The application chooses random inputs every cycle.
The network is a single layered neural network that feeds back into itself.

Currently the network dies out due to purposeful neuron degregation meant to "erase" weights that aren't used frequently.
The size of the network is small due to limitation of the Device namespace. If I were to have used something else other than implementing it myself I'm sure networks of much larger sizes can be made.

The idea behind this application is to demonstrate a slow learning technique that uses a concept base around artificial selection.

# Compiling
You must have a CUDA compatible graphics card to link against the required CUDA libraries that are used.
The application was written in C++ but all files are to be complied with the NVCC (Nvidia Cuda Compiler).
