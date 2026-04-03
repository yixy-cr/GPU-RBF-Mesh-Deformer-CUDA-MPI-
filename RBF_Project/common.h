#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Represents a control point with its original position and computed displacement
struct ControlPoint {
    double x, y, z;      // Original coordinates
    double dx, dy, dz;   // Displacement vector
    double p, s00;       // Physical parameters from input file
};

// Represents a node from the main mesh with its original position and computed displacement
struct MeshNode {
    int id;
    double x, y, z;      // Original coordinates
    double dx, dy, dz;   // Resulting displacement vector (to be computed)
};

#endif // COMMON_H

