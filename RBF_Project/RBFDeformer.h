#ifndef RBF_DEFORMER_H
#define RBF_DEFORMER_H

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>

// Optional cuSOLVERMp integration. Define USE_CUSOLVERMP when building on a
// system that provides cusolverMp.h and the NVHPC CAL libraries.
#ifdef USE_CUSOLVERMP
#include <cusolverMp.h>
#endif
#include "common.h"

class RBFDeformer {
public:
    enum class SolveBackend {
        None = 0,
        CuSolverMpDistributedLU,
        RootCuSolverDnFallback,
        SingleCuSolverDn
    };

    RBFDeformer();
    ~RBFDeformer();

    /**
     * @brief Initializes the deformer, allocates GPU memory, and copies data.
     * @param mesh The mesh nodes (evaluation points).
     * @param cps The control points.
     * @param radius The support radius for the RBF function.
     */
    void init(const std::vector<MeshNode>& mesh, const std::vector<ControlPoint>& cps, double radius);

    /**
     * @brief Executes the main deformation logic on the GPU.
     * 
     * This involves building the RBF matrix, solving the linear system for
     * coefficients using cuSOLVER, and then interpolating the displacements
     * for all mesh nodes.
     */
    void deform();

    /**
     * @brief Copies the computed displacements from the GPU back to the host.
     * @param outNodes The vector to receive the results. The dx, dy, dz fields will be updated.
     */
    void getResults(std::vector<MeshNode>& outNodes);

    // --- Distributed interfaces ---
    // Initialize any distributed solver state (e.g. cusolverMp, CAL, comm grid)
    void init_distributed(MPI_Comm comm = MPI_COMM_WORLD);

    // Solve the RBF linear system in a distributed fashion using cusolverMp.
    // This is a high-level entry point that will perform any necessary
    // local matrix construction, call the cusolverMp routines and ensure
    // that the full coefficient vector is available on each rank after return.
    void solve_distributed(const std::vector<ControlPoint>& global_cps, MPI_Comm comm = MPI_COMM_WORLD);

    // Interpolate displacements for a local subset of mesh nodes using
    // globally-available control points and coefficients.
    void interpolate_distributed(std::vector<MeshNode>& local_mesh, const std::vector<ControlPoint>& cps, MPI_Comm comm = MPI_COMM_WORLD);

    // Host <-> device coefficient helpers used for broadcasting coefficients
    void getCoefficientsHost(std::vector<double>& out);
    void setCoefficientsFromHost(const std::vector<double>& in, int nb_in);
    // Set support radius (host-side) used by interpolate kernels (radius squared stored)
    void setSupportRadius(double radius);

    // --- Run-mode introspection (for logging / reproducibility) ---
    SolveBackend getLastSolveBackend() const { return last_solve_backend; }
    const char* getLastSolveBackendName() const;
    bool lastSolveWasDistributed() const { return last_solve_backend == SolveBackend::CuSolverMpDistributedLU; }
private:
    double supportRadiusSq; // Store r^2 for faster comparison
    int nb; // Number of control points
    int ni; // Number of mesh nodes (interior points)
    int matrix_dim; // Dimension of the square matrix A (nb + 4)

    // Device pointers
    double* d_CP_pos;      // Control Points positions (x, y, z)
    double* d_CP_disp;     // Control Points displacements (dx, dy, dz)
    double* d_Mesh_pos;    // Mesh Nodes positions (x, y, z)
    double* d_Mesh_disp;   // Mesh Nodes displacements (dx, dy, dz) - result
    double* d_MatrixA;     // RBF System Matrix A
    double* d_Rhs_b;       // Right-hand side vector 'b' (becomes solution 'x' in place)
    int* d_Pivots;         // Pivot array for LU decomposition
    int* d_Info;           // Status from cuSOLVER (for singularity check)
    double* d_Work;        // Workspace for cuSOLVER
    int lwork;             // Size of the workspace

    // Host-side pointer for singularity check
    int h_Info = 0;

    cusolverDnHandle_t cusolverH;
    // Distributed solver handle
#ifdef USE_CUSOLVERMP
    cusolverMpHandle_t cusolverMpH = nullptr;
#else
    // placeholder when cuSOLVERMp isn't available at compile time
    void* cusolverMpH = nullptr;
#endif

    // Process grid (2D) for block-cyclic distribution
    int proc_rows = 1;
    int proc_cols = 1;
    int block_size = 64; // default block size for block-cyclic distribution

    // MPI info
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    int mpi_rank = 0;
    int mpi_size = 1;
    // Device pointer for full coefficients (matrix_dim x 3) stored contiguously
    double* d_Coeffs = nullptr;
    // Optional device-local buffer holding the local block of A in cusolverMp layout
    double* d_localAroot_device = nullptr;
    size_t d_localAroot_elems = 0;
    int d_localAroot_rows = 0;
    int d_localAroot_cols = 0;

    // Configure a 2D process grid for block-cyclic distribution (pr x pc)
    void configure_process_grid(int pr, int pc, int blk_size = 64);

    // Compute the list of global row indices owned by this rank given block-cyclic
    // distribution. Returns a host vector of owned global row indices.
    std::vector<int> compute_owned_global_rows(int nrows) const;

    // Attempt to solve using cuSOLVERMp. Returns true on success; false means
    // caller should fall back to the centralized solve path. This function is
    // implemented as a conditional stub when cuSOLVERMp is not available.
    bool try_cusolvermp_solve(const std::vector<int>& owned_rows, const std::vector<double>& h_localA, int local_rows, const std::vector<double>& h_localB, int n);

    // Tracks whether d_Coeffs contains valid coefficients for interpolation.
    bool coeffs_ready = false;
    int coeffs_matrix_dim = 0;

    // Tracks which solve path actually ran last.
    SolveBackend last_solve_backend = SolveBackend::None;
};

#endif // RBF_DEFORMER_H
