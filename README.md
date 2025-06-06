# A template for Mojo CPU / GPU custom operation development #

This project is a work in progress towards implementing a Non Maximum Suppression kernel. This work
has started during the Mojo Hackathon in May 2025 based on a template provided by Modular.

Only the `magic run test` is used at the moment.

## Algorithm

The input to the module are two tensors:
- the corners of the bounding boxes with shape N, 4 (x\_low, y\_low, x\_high, y\_high)
- the scores for each box

There is also a parameter named iou\_threshold, where iou is Intersection Over Union.

For the bounding boxes that overlap sufficiently, i.e. iou > iou\_threshold, only the bounding box with the 
higher score will be kept.

## Current status

A keep\_bitmap tensor of uint8 with shape N is returned. The value will be 1 for the bounding boxes to keep, 0 to discard.

## Running tests ##

There are both Mojo unit tests and `pytest`-based Python unit tests in this
template. If you're working from pure Mojo code, you can start with the former,
and can move to the Python-based tests if you'd like to verify an operation
inside of a Python MAX graph.

To run the Mojo unit tests, use

```sh
magic run test
```

To run the `pytest` unit tests, use

```sh
magic run pytest
```

## Benchmarking ##

A series of rigorous performance benchmarks for the Mojo kernel in development
have been configured using the Mojo `benchmark` module. To run them, use the
command

```sh
magic run benchmarks
```

## Running a graph containing this operation ##

A very basic one-operation graph in Python that contains the custom Mojo
operation can be run using

```sh
magic run graph
```

This will compile the Mojo code for the kernel, place it in a graph, compile
the graph, and run it. Such a graph is an example of how this operation would
be used in a larger AI model within the MAX framework.

## Profiling AMD GPU kernels ##

Run the command:

```sh
magic run profile_amd test_correctness.mojo
```

This will generate a `test_correctness.log` file containing profile information.

Check the script inside [profile_amd.sh](./profile_amd.sh) to see how it works.

## Debugging AMD GPU kernels ##

To build a binary from a Mojo file and start the debugger run:

```sh
magic run debug_amd test_correctness.mojo
```

You can now set a breakpoint inside the kernel code (press y on the "Make
breakpoint pending" prompt):

```sh
b operations/matrix_multiplication.mojo:180
```

Cheat-sheet for debugging commands:

```sh
# Start the debug session
run
r

# Continue to next break point
continue
c

# Step over
next
n

# Step into
step
s

# List all host and GPU threads
info threads

# Switch to thread n
thread [n]

# View local variables
info locals

# view register values
info register

# view all the stack frames
backtrace
bt

# switch to frame n
frame [n]
```

For more commands run:

```sh
help
help [topic]
```

Check the script inside [debug_amd.sh](./debug_amd.sh) to see how it works.

## License ##

Apache License v2.0 with LLVM Exceptions.
