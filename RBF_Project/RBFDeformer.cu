#include "RBFDeformer.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <limits>
#include <cstdlib>

#ifdef USE_CUSOLVERMP
#include <nccl.h>
#include <cal.h>
#endif

#ifdef USE_CUSOLVERMP
// Simple CAL communicator backend implemented on top of MPI collectives.
// cuSOLVERMp requires a cal_comm_t; CAL itself lets us supply transport
// callbacks. We use blocking MPI_Allgather for correctness (sufficient for
// modest matrix sizes); performance tuning can later replace this with UCC/UCX.
struct CalMPIContext {
    MPI_Comm comm;
};

static calError_t cal_mpi_allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request) {
    CalMPIContext* ctx = reinterpret_cast<CalMPIContext*>(data);
    if (!ctx) return CAL_ERROR_INVALID_PARAMETER;
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) return CAL_ERROR_INVALID_PARAMETER;
    int count = static_cast<int>(size);
    MPI_Allgather(src_buf, count, MPI_BYTE, recv_buf, count, MPI_BYTE, ctx->comm);
    if (request) *request = nullptr; // synchronous completion
    return CAL_OK;
}

static calError_t cal_mpi_req_test(void* /*request*/) {
    // Blocking allgather above completes immediately; nothing to test.
    return CAL_OK;
}

static calError_t cal_mpi_req_free(void* /*request*/) {
    return CAL_OK;
}
#endif

// Wendland C4 radial basis helper (host+device)
static __host__ __device__ inline double wendland_c4_rbf(double dist_sq, double r_sq) {
    if (dist_sq >= r_sq) return 0.0;
    double xi = sqrt(dist_sq / r_sq);
    double one_minus_xi = 1.0 - xi;
    double term1 = one_minus_xi * one_minus_xi * one_minus_xi;
    term1 = term1 * term1;
    double term2 = (35.0 / 3.0) * xi * xi + 6.0 * xi + 1.0;
    return term1 * term2;
}

// Forward declaration of CUDA kernels
__global__ void build_rbf_matrix_kernel(double* matrix, const double* cp_pos, int nb, double r_sq);
__global__ void interpolate_displacement_kernel(const double* mesh_pos, double* mesh_disp, const double* cp_pos, const double* coeffs, int ni, int nb, double r_sq);

// Build local rows kernel: each block/thread computes one entry of a local row
// localA is row-major with dimensions local_rows x ncols (ncols == matrix_dim)
// col_offset: global column index offset for this block
// block_ncols: number of columns in this local block
__global__ void build_rbf_localrows_kernel(double* localA, const double* cp_pos, const int* global_rows, int local_rows, int ncols, int nb, double r_sq, int col_offset, int block_ncols, int lld) {
    int lr = blockIdx.y * blockDim.y + threadIdx.y; // local row index
    int col_in_block = blockIdx.x * blockDim.x + threadIdx.x; // index within this block
    if (lr >= local_rows || col_in_block >= block_ncols) return;

    int gi = global_rows[lr]; // global row index
    int gj = col_offset + col_in_block; // global column index
    double val = 0.0;
    if (gi < nb && gj < nb) {
        double dx = cp_pos[gi * 3 + 0] - cp_pos[gj * 3 + 0];
        double dy = cp_pos[gi * 3 + 1] - cp_pos[gj * 3 + 1];
        double dz = cp_pos[gi * 3 + 2] - cp_pos[gj * 3 + 2];
        double dist_sq = dx * dx + dy * dy + dz * dz;
        // reuse Wendland function
        if (dist_sq < r_sq) {
            double xi = sqrt(dist_sq / r_sq);
            double one_minus_xi = 1.0 - xi;
            double term1 = one_minus_xi * one_minus_xi * one_minus_xi;
            term1 = term1 * term1;
            double term2 = (35.0 / 3.0) * xi * xi + 6.0 * xi + 1.0;
            val = term1 * term2;
        } else {
            val = 0.0;
        }
    } else if (gi < nb && gj >= nb) {
        // polynomial part
        if (gj == nb) val = 1.0;
        else if (gj == nb + 1) val = cp_pos[gi * 3 + 0];
        else if (gj == nb + 2) val = cp_pos[gi * 3 + 1];
        else if (gj == nb + 3) val = cp_pos[gi * 3 + 2];
    } else if (gi >= nb && gj < nb) {
        if (gi == nb) val = 1.0;
        else if (gi == nb + 1) val = cp_pos[gj * 3 + 0];
        else if (gi == nb + 2) val = cp_pos[gj * 3 + 1];
        else if (gi == nb + 3) val = cp_pos[gj * 3 + 2];
    }

    // localA uses column-major layout with leading dimension lld (= local_rows)
    localA[(size_t)(col_offset + col_in_block) * lld + (size_t)lr] = val;
}

// Interpolate kernel that uses global coefficients (coeffs is matrix_dim x 3)
__global__ void interpolate_local_kernel(const double* mesh_pos, double* mesh_disp, const double* cp_pos, const double* coeffs, int ni, int nb, int matrix_dim, double r_sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ni) return;

    double disp_x = 0.0, disp_y = 0.0, disp_z = 0.0;
    double node_x = mesh_pos[i * 3 + 0];
    double node_y = mesh_pos[i * 3 + 1];
    double node_z = mesh_pos[i * 3 + 2];

    const double* a_x = coeffs;
    const double* a_y = coeffs + matrix_dim;
    const double* a_z = coeffs + 2 * matrix_dim;

    for (int j = 0; j < nb; ++j) {
        double c_x = cp_pos[j * 3 + 0];
        double c_y = cp_pos[j * 3 + 1];
        double c_z = cp_pos[j * 3 + 2];
        double dx = node_x - c_x;
        double dy = node_y - c_y;
        double dz = node_z - c_z;
        double dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq < r_sq) {
            double xi = sqrt(dist_sq / r_sq);
            double one_minus_xi = 1.0 - xi;
            double term1 = one_minus_xi * one_minus_xi * one_minus_xi;
            term1 = term1 * term1;
            double term2 = (35.0 / 3.0) * xi * xi + 6.0 * xi + 1.0;
            double phi = term1 * term2;
            disp_x += a_x[j] * phi;
            disp_y += a_y[j] * phi;
            disp_z += a_z[j] * phi;
        }
    }
    // polynomial part
    disp_x += a_x[nb] + a_x[nb+1]*node_x + a_x[nb+2]*node_y + a_x[nb+3]*node_z;
    disp_y += a_y[nb] + a_y[nb+1]*node_x + a_y[nb+2]*node_y + a_y[nb+3]*node_z;
    disp_z += a_z[nb] + a_z[nb+1]*node_x + a_z[nb+2]*node_y + a_z[nb+3]*node_z;

    mesh_disp[i * 3 + 0] = disp_x;
    mesh_disp[i * 3 + 1] = disp_y;
    mesh_disp[i * 3 + 2] = disp_z;
}

// Distributed helper forward declarations (placeholders for now)
void build_local_block_matrix_placeholder();

// --- RBFDeformer Class Implementation ---

RBFDeformer::RBFDeformer() : cusolverH(nullptr), d_CP_pos(nullptr), d_CP_disp(nullptr),
                           d_Mesh_pos(nullptr), d_Mesh_disp(nullptr), d_MatrixA(nullptr),
                           d_Rhs_b(nullptr), d_Pivots(nullptr), d_Info(nullptr), d_Work(nullptr), lwork(0) {
}


RBFDeformer::~RBFDeformer() {
    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (d_CP_pos) CUDA_CHECK(cudaFree(d_CP_pos));
    if (d_CP_disp) CUDA_CHECK(cudaFree(d_CP_disp));
    if (d_Mesh_pos) CUDA_CHECK(cudaFree(d_Mesh_pos));
    if (d_Mesh_disp) CUDA_CHECK(cudaFree(d_Mesh_disp));
    if (d_MatrixA) CUDA_CHECK(cudaFree(d_MatrixA));
    if (d_Rhs_b) CUDA_CHECK(cudaFree(d_Rhs_b));
    if (d_Pivots) CUDA_CHECK(cudaFree(d_Pivots));
    if (d_Info) CUDA_CHECK(cudaFree(d_Info));
    if (d_Work) CUDA_CHECK(cudaFree(d_Work));
    if (d_Coeffs) CUDA_CHECK(cudaFree(d_Coeffs));
}

void RBFDeformer::init_distributed(MPI_Comm comm) {
    mpi_comm = comm;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    // Create cusolverMp handle if compiled with support. Actual process-grid and
    // CAL initialization will be implemented when NVHPC libs are available.
#ifdef USE_CUSOLVERMP
    // Defer actual cuSOLVERMp handle creation to the full implementation in
    // `try_cusolvermp_solve`, because proper initialization requires CAL
    // communicators and additional arguments whose availability depends on
    // the target system. For now set handle to null and warn so the fallback
    // central-solve path remains functional.
    cusolverMpH = nullptr;
    if (mpi_rank == 0) std::cerr << "提示：已编译启用 cuSOLVERMp；句柄创建延后到实际求解阶段。" << std::endl;
#else
    // No cuSOLVERMp available at compile time; leave cusolverMpH as null.
    cusolverMpH = nullptr;
#endif
}

void RBFDeformer::configure_process_grid(int pr, int pc, int blk_size) {
    if (pr <= 0) pr = 1;
    if (pc <= 0) pc = 1;
    proc_rows = pr;
    proc_cols = pc;
    block_size = blk_size > 0 ? blk_size : block_size;
    if (mpi_rank == 0) {
        std::cout << "已配置进程网格：" << proc_rows << " x " << proc_cols << "，block_size=" << block_size << std::endl;
    }
}

std::vector<int> RBFDeformer::compute_owned_global_rows(int nrows) const {
    std::vector<int> rows;
    if (nrows <= 0) return rows;
    int bs = block_size;
    int nblocks = (nrows + bs - 1) / bs;
    // choose row coordinate for this rank in implicit row-major rank ordering
    int myRowCoord = mpi_rank % proc_rows;
    for (int br = 0; br < nblocks; ++br) {
        if ((br % proc_rows) == myRowCoord) {
            int start = br * bs;
            int block_sz = std::min(bs, nrows - start);
            for (int t = 0; t < block_sz; ++t) rows.push_back(start + t);
        }
    }
    return rows;
}

void RBFDeformer::solve_distributed(const std::vector<ControlPoint>& global_cps, MPI_Comm comm) {
    // Placeholder / fallback implementation for distributed solve.
    // Full cusolverMp-based distributed factorization will be implemented
    // in subsequent patches. For now, if mpi_size == 1 we call the local
    // path; if more than 1, we currently let rank 0 perform the solve and
    // (future) code will distribute building and solving.
    if (mpi_comm == MPI_COMM_NULL) {
        mpi_comm = comm;
        MPI_Comm_rank(mpi_comm, &mpi_rank);
        MPI_Comm_size(mpi_comm, &mpi_size);
    }

    if (mpi_size <= 1) {
        if (mpi_rank == 0) {
            // Single-rank case: caller will use local init/deform path.
        }
        MPI_Barrier(mpi_comm);
        return;
    }

    // Ensure nb and related dimensions are initialized from caller-provided control points
    nb = (int)global_cps.size();
    matrix_dim = nb + 4;
    // Mark coefficients invalid until a solve path succeeds.
    coeffs_ready = false;
    coeffs_matrix_dim = matrix_dim;
    last_solve_backend = SolveBackend::None;
    if (supportRadiusSq <= 0.0) {
        // default fallback support radius (matches main.cu default 0.01)
        supportRadiusSq = 0.01 * 0.01;
    }
    std::cerr << "[solve_distributed] 进入：rank=" << mpi_rank << " mpi_size=" << mpi_size << " nb=" << nb << std::endl;
    auto log_mem = [&](const char* stage) {
        size_t free_b = 0, total_b = 0;
        cudaError_t st = cudaMemGetInfo(&free_b, &total_b);
        if (st == cudaSuccess) {
            std::cerr << "[mem] rank=" << mpi_rank << " " << stage
                      << " free=" << free_b << " total=" << total_b << std::endl;
        } else {
            std::cerr << "[mem] rank=" << mpi_rank << " " << stage
                      << " cudaMemGetInfo failed: " << cudaGetErrorString(st) << std::endl;
        }
    };
    log_mem("enter_solve_distributed");

    // For the first working distributed implementation we implement a row-distributed
    // block-cyclic builder: each rank computes a subset of full rows of the global
    // matrix A (rows are distributed in block-cyclic manner). Each rank computes
    // its local rows (all columns) on the GPU, then we gather the rows on rank0,
    // perform the solve on rank0 (using cusolverDn as before), and broadcast the
    // solution coefficients to all ranks. This provides distributed assembly with
    // a central solve; replacing the central solve with cusolverMp will be a
    // subsequent step.

    int n = nb + 4;
    int bs = block_size > 0 ? block_size : 64; // block size for block-cyclic distribution

    // True 1D block-cyclic row ownership to match cuSOLVERMp grid pr=mpi_size, pc=1
    std::vector<int> global_rows;
    global_rows.reserve(n / mpi_size + bs);
    int nblocks = (n + bs - 1) / bs;
    for (int br = 0; br < nblocks; ++br) {
        if ((br % mpi_size) != mpi_rank) continue;
        int start = br * bs;
        int bsz = std::min(bs, n - start);
        for (int t = 0; t < bsz; ++t) global_rows.push_back(start + t);
    }
    int local_rows = (int)global_rows.size();
    // Each local row has n columns

    // Prepare cp_pos on device
    std::vector<double> h_cp_pos(nb * 3);
    for (int i = 0; i < nb; ++i) {
        h_cp_pos[i*3+0] = global_cps[i].x;
        h_cp_pos[i*3+1] = global_cps[i].y;
        h_cp_pos[i*3+2] = global_cps[i].z;
    }
    double* d_cp_pos_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cp_pos_local, nb * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cp_pos_local, h_cp_pos.data(), nb * 3 * sizeof(double), cudaMemcpyHostToDevice));
    log_mem("after_alloc_cp_pos");

    // Allocate device buffer for global_rows indices
    int* d_global_rows = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global_rows, local_rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_global_rows, global_rows.data(), local_rows * sizeof(int), cudaMemcpyHostToDevice));
    log_mem("after_alloc_global_rows");

    // Build localA directly into a device-local buffer (avoid root gather)
    double* d_localAroot_device = nullptr;
    size_t d_localAroot_bytes = (size_t)local_rows * (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_localAroot_device, d_localAroot_bytes));
    CUDA_CHECK(cudaMemset(d_localAroot_device, 0, d_localAroot_bytes));
    log_mem("after_alloc_localA");
    if (mpi_rank == 0) std::cerr << "[debug] rank=" << mpi_rank << " allocated d_localAroot_device bytes=" << d_localAroot_bytes << std::endl;
    // determine block columns based on available device memory
    size_t free_mem = 0, total_mem = 0;
    cudaError_t memerr = cudaMemGetInfo(&free_mem, &total_mem);
    int block_ncols = 4096; // default
    if (memerr == cudaSuccess && free_mem > 0) {
        // leave some headroom (use 60% of free mem)
        size_t usable = (size_t)(free_mem * 0.6);
        // block_ncols <= usable / (local_rows * sizeof(double))
        size_t max_cols = usable / ((size_t)local_rows * sizeof(double));
        if (max_cols >= 1) {
            block_ncols = (int)std::min((size_t)n, max_cols);
            // limit block size to reasonable chunk
            block_ncols = std::max(256, std::min(block_ncols, 16384));
        }
    }

    int col_offset = 0;
    dim3 block(32, 8);
    for (col_offset = 0; col_offset < n; col_offset += block_ncols) {
        int this_ncols = std::min(block_ncols, n - col_offset);
        dim3 grid((this_ncols + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);
        // launch kernel to write block directly into the device-local A buffer
        build_rbf_localrows_kernel<<<grid, block>>>(d_localAroot_device, d_cp_pos_local, d_global_rows, local_rows, n, nb, supportRadiusSq, col_offset, this_ncols, local_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::cerr << "[debug] rank=" << mpi_rank << " finished block-wise assembly (local_rows=" << local_rows << ", n=" << n << ")" << std::endl;
    log_mem("after_assembly_localA");

    // Build local RHS rows (b) for control point displacements: three columns (dx,dy,dz)
    std::vector<double> h_localB((size_t)local_rows * 3, 0.0);
    for (int lr = 0; lr < local_rows; ++lr) {
        int gi = global_rows[lr];
        if (gi < nb) {
            h_localB[lr * 3 + 0] = global_cps[gi].dx;
            h_localB[lr * 3 + 1] = global_cps[gi].dy;
            h_localB[lr * 3 + 2] = global_cps[gi].dz;
        } else {
            h_localB[lr * 3 + 0] = 0.0;
            h_localB[lr * 3 + 1] = 0.0;
            h_localB[lr * 3 + 2] = 0.0;
        }
    }

    // First, try to use cuSOLVERMp distributed solver (conditional).
    // If RBF_REQUIRE_CUSOLVERMP is set, failure will be treated as fatal (no fallback).
    bool cusolvermp_ok = false;
    bool disable_cusolvermp = (std::getenv("RBF_DISABLE_CUSOLVERMP") != nullptr);
    bool require_cusolvermp = (std::getenv("RBF_REQUIRE_CUSOLVERMP") != nullptr);
    std::vector<double> dummy_localB; // will be populated below if needed
#ifdef USE_CUSOLVERMP
    // expose device-local A to try_cusolvermp_solve to allow D2D usage
    this->d_localAroot_device = d_localAroot_device;
    this->d_localAroot_elems = (size_t)local_rows * (size_t)n;
    this->d_localAroot_rows = local_rows;
    this->d_localAroot_cols = n;
#endif
#ifdef USE_CUSOLVERMP
    if (!disable_cusolvermp) {
        // Prepare host RHS for local rows
        dummy_localB.resize((size_t)local_rows * 3);
        for (int lr = 0; lr < local_rows; ++lr) {
            int gi = global_rows[lr];
            if (gi < nb) {
                dummy_localB[lr * 3 + 0] = global_cps[gi].dx;
                dummy_localB[lr * 3 + 1] = global_cps[gi].dy;
                dummy_localB[lr * 3 + 2] = global_cps[gi].dz;
            } else {
                dummy_localB[lr * 3 + 0] = 0.0;
                dummy_localB[lr * 3 + 1] = 0.0;
                dummy_localB[lr * 3 + 2] = 0.0;
            }
        }
        log_mem("before_try_cusolvermp");
        std::cerr << "[distributed] rank=" << mpi_rank << " 尝试 cuSOLVERMp 求解（local_rows=" << local_rows << "）" << std::endl;
        cusolvermp_ok = try_cusolvermp_solve(global_rows, std::vector<double>{}, local_rows, dummy_localB, n);
        std::cerr << "[debug] rank=" << mpi_rank << " try_cusolvermp_solve returned " << (cusolvermp_ok?"OK":"FAIL") << std::endl;
    } else if (mpi_rank == 0) {
        std::cerr << "[distributed] 已通过 RBF_DISABLE_CUSOLVERMP 禁用 cuSOLVERMp 路径。" << std::endl;
    }
#else
    if (require_cusolvermp) {
        if (mpi_rank == 0) {
            std::cerr << "[distributed] 设置了 RBF_REQUIRE_CUSOLVERMP，但该程序未以 USE_CUSOLVERMP 编译。" << std::endl;
        }
        MPI_Abort(mpi_comm, 2);
    }
#endif

    // Ensure all ranks agree on cuSOLVERMp success/failure.
    int cusolvermp_ok_i = cusolvermp_ok ? 1 : 0;
    int cusolvermp_ok_all = 0;
    MPI_Allreduce(&cusolvermp_ok_i, &cusolvermp_ok_all, 1, MPI_INT, MPI_MIN, mpi_comm);
    std::cerr << "[debug] rank=" << mpi_rank << " cusolvermp_ok_i=" << cusolvermp_ok_i << " allreduce-> " << cusolvermp_ok_all << std::endl;

    if (require_cusolvermp) {
        if (disable_cusolvermp) {
            if (mpi_rank == 0) {
                std::cerr << "[distributed] 设置了 RBF_REQUIRE_CUSOLVERMP，但又通过 RBF_DISABLE_CUSOLVERMP 禁用了 cuSOLVERMp。" << std::endl;
            }
            MPI_Abort(mpi_comm, 2);
        }
        if (cusolvermp_ok_all == 0) {
            if (mpi_rank == 0) {
                std::cerr << "[distributed] cuSOLVERMp 求解失败，且设置了 RBF_REQUIRE_CUSOLVERMP；终止运行（不允许回退）。" << std::endl;
            }
            MPI_Abort(mpi_comm, 2);
        }
    }
    // If cuSOLVERMp succeeded, return directly (fully distributed path).
    if (cusolvermp_ok_all == 1) {
        if (mpi_rank == 0) {
            std::cerr << "[distributed] cuSOLVERMp 求解成功；已完成全分布式求解。" << std::endl;
        }
        // coefficients are already broadcast/copyed into d_Coeffs inside try_cusolvermp_solve
        coeffs_ready = true;
        coeffs_matrix_dim = n;
        last_solve_backend = SolveBackend::CuSolverMpDistributedLU;
        CUDA_CHECK(cudaFree(d_cp_pos_local));
        CUDA_CHECK(cudaFree(d_global_rows));
        // free device-local A buffer if we allocated it here
        if (this->d_localAroot_device) {
            CUDA_CHECK(cudaFree(this->d_localAroot_device));
            this->d_localAroot_device = nullptr;
            this->d_localAroot_elems = 0;
            this->d_localAroot_rows = 0;
            this->d_localAroot_cols = 0;
        }
        MPI_Barrier(mpi_comm);
        return;
    }

    // Fully distributed mode disallows root fallback solve.
    if (mpi_rank == 0) {
        std::cerr << "[distributed] cuSOLVERMp 求解失败；当前模式禁止 root 回退，终止运行。" << std::endl;
    }
    CUDA_CHECK(cudaFree(d_cp_pos_local));
    CUDA_CHECK(cudaFree(d_global_rows));
    if (d_localAroot_device) {
        CUDA_CHECK(cudaFree(d_localAroot_device));
    }
    MPI_Abort(mpi_comm, 2);

}

// Original full-matrix builder (single-node path). Keeps compatibility with
// the non-distributed `deform()` code path which expects a full matrix build.
__global__ void build_rbf_matrix_kernel(double* matrix, const double* cp_pos, int nb, double r_sq) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nb + 4;

    if (row >= n || col >= n) return;

    size_t index = (size_t)row * (size_t)n + (size_t)col;
    double val = 0.0;

    if (row < nb && col < nb) {
        double dx = cp_pos[row * 3 + 0] - cp_pos[col * 3 + 0];
        double dy = cp_pos[row * 3 + 1] - cp_pos[col * 3 + 1];
        double dz = cp_pos[row * 3 + 2] - cp_pos[col * 3 + 2];
        double dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq < r_sq) {
            double xi = sqrt(dist_sq / r_sq);
            double one_minus_xi = 1.0 - xi;
            double term1 = one_minus_xi * one_minus_xi * one_minus_xi;
            term1 = term1 * term1;
            double term2 = (35.0 / 3.0) * xi * xi + 6.0 * xi + 1.0;
            val = term1 * term2;
        } else {
            val = 0.0;
        }

    } else if (row < nb && col >= nb) {
        if (col == nb) val = 1.0;
        else if (col == nb + 1) val = cp_pos[row * 3 + 0]; // x
        else if (col == nb + 2) val = cp_pos[row * 3 + 1]; // y
        else if (col == nb + 3) val = cp_pos[row * 3 + 2]; // z
    } else if (row >= nb && col < nb) {
        if (row == nb) val = 1.0;
        else if (row == nb + 1) val = cp_pos[col * 3 + 0]; // x
        else if (row == nb + 2) val = cp_pos[col * 3 + 1]; // y
        else if (row == nb + 3) val = cp_pos[col * 3 + 2]; // z
    }

    matrix[index] = val;
}

// Add a small regularization to the diagonal: A[i*n + i] += eps
__global__ void add_diag_kernel(double* A, int n, double eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[(size_t)i * n + i] += eps;
}

__global__ void interpolate_displacement_kernel(const double* mesh_pos, double* mesh_disp, const double* cp_pos, const double* coeffs, int ni, int nb, double r_sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ni) return;

    double disp_x = 0.0, disp_y = 0.0, disp_z = 0.0;
    double node_x = mesh_pos[i * 3 + 0];
    double node_y = mesh_pos[i * 3 + 1];
    double node_z = mesh_pos[i * 3 + 2];

    int matrix_dim = nb + 4;
    const double* a_x = coeffs;
    const double* a_y = coeffs + matrix_dim;
    const double* a_z = coeffs + 2*matrix_dim;

    // --- RBF Summation Part ---
    for (int j = 0; j < nb; ++j) {
        double c_x = cp_pos[j * 3 + 0];
        double c_y = cp_pos[j * 3 + 1];
        double c_z = cp_pos[j * 3 + 2];

        double dx = node_x - c_x;
        double dy = node_y - c_y;
        double dz = node_z - c_z;
        double dist_sq = dx * dx + dy * dy + dz * dz;

        double phi = wendland_c4_rbf(dist_sq, r_sq);
        
        if (phi > 0) {
            disp_x += a_x[j] * phi;
            disp_y += a_y[j] * phi;
            disp_z += a_z[j] * phi;
        }
    }

    // --- Polynomial Part ---
    disp_x += a_x[nb] + a_x[nb+1]*node_x + a_x[nb+2]*node_y + a_x[nb+3]*node_z;
    disp_y += a_y[nb] + a_y[nb+1]*node_x + a_y[nb+2]*node_y + a_y[nb+3]*node_z;
    disp_z += a_z[nb] + a_z[nb+1]*node_x + a_z[nb+2]*node_y + a_z[nb+3]*node_z;
    
    mesh_disp[i * 3 + 0] = disp_x;
    mesh_disp[i * 3 + 1] = disp_y;
    mesh_disp[i * 3 + 2] = disp_z;
}

void RBFDeformer::interpolate_distributed(std::vector<MeshNode>& local_mesh, const std::vector<ControlPoint>& cps, MPI_Comm comm) {
    if (mpi_comm == MPI_COMM_NULL) {
        mpi_comm = comm;
        MPI_Comm_rank(mpi_comm, &mpi_rank);
        MPI_Comm_size(mpi_comm, &mpi_size);
    }

    int local_ni = (int)local_mesh.size();
    if (local_ni == 0) return;

    int nb_local = (int)cps.size();
    // Prepare cp_pos on host
    std::vector<double> h_cp_pos(nb_local * 3);
    for (int i = 0; i < nb_local; ++i) {
        h_cp_pos[i*3+0] = cps[i].x;
        h_cp_pos[i*3+1] = cps[i].y;
        h_cp_pos[i*3+2] = cps[i].z;
    }
    double* d_cp_pos_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cp_pos_local, nb_local * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cp_pos_local, h_cp_pos.data(), nb_local * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Prepare mesh pos on device
    std::vector<double> h_mesh_pos(local_ni * 3);
    for (int i = 0; i < local_ni; ++i) {
        h_mesh_pos[i*3+0] = local_mesh[i].x;
        h_mesh_pos[i*3+1] = local_mesh[i].y;
        h_mesh_pos[i*3+2] = local_mesh[i].z;
    }

    double* d_mesh_pos_local = nullptr;
    double* d_mesh_disp_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mesh_pos_local, local_ni * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mesh_disp_local, local_ni * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_mesh_pos_local, h_mesh_pos.data(), local_ni * 3 * sizeof(double), cudaMemcpyHostToDevice));

    int matrix_dim = nb_local + 4;
    if (!coeffs_ready || !d_Coeffs) {
        throw std::runtime_error("Coefficients not available/valid for interpolation (distributed solve failed or not run). Set RBF_DISABLE_CUSOLVERMP=1 to force fallback solve.");
    }
    if (coeffs_matrix_dim != 0 && coeffs_matrix_dim != matrix_dim) {
        throw std::runtime_error("Coefficient dimension mismatch for interpolation (coeffs_matrix_dim != nb_local+4). This usually indicates an earlier solve failure or mixed control-point sets.");
    }
    int block = 256;
    int grid = (local_ni + block - 1) / block;
    interpolate_local_kernel<<<grid, block>>>(d_mesh_pos_local, d_mesh_disp_local, d_cp_pos_local, d_Coeffs, local_ni, nb_local, matrix_dim, supportRadiusSq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_mesh_disp(local_ni * 3);
    CUDA_CHECK(cudaMemcpy(h_mesh_disp.data(), d_mesh_disp_local, local_ni * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < local_ni; ++i) {
        local_mesh[i].dx = h_mesh_disp[i*3+0];
        local_mesh[i].dy = h_mesh_disp[i*3+1];
        local_mesh[i].dz = h_mesh_disp[i*3+2];
    }

    CUDA_CHECK(cudaFree(d_cp_pos_local));
    CUDA_CHECK(cudaFree(d_mesh_pos_local));
    CUDA_CHECK(cudaFree(d_mesh_disp_local));
}

// --- Single-node API implementations (init / deform / getResults) ---
void RBFDeformer::init(const std::vector<MeshNode>& mesh, const std::vector<ControlPoint>& cps, double radius) {
    // Ensure cuSOLVER handle exists (create lazily to avoid ctor exceptions)
    if (cusolverH == nullptr) {
        cusolverStatus_t status = cusolverDnCreate(&cusolverH);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSOLVER handle.");
        }
    }
    nb = (int)cps.size();
    ni = (int)mesh.size();
    matrix_dim = nb + 4;
    supportRadiusSq = radius * radius;

    // Copy control points positions and displacements to device
    if (d_CP_pos) CUDA_CHECK(cudaFree(d_CP_pos));
    if (d_CP_disp) CUDA_CHECK(cudaFree(d_CP_disp));
    CUDA_CHECK(cudaMalloc(&d_CP_pos, (size_t)nb * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_CP_disp, (size_t)nb * 3 * sizeof(double)));
    std::vector<double> h_cp_pos(nb * 3), h_cp_disp(nb * 3, 0.0);
    for (int i = 0; i < nb; ++i) {
        h_cp_pos[i*3 + 0] = cps[i].x;
        h_cp_pos[i*3 + 1] = cps[i].y;
        h_cp_pos[i*3 + 2] = cps[i].z;
        h_cp_disp[i*3 + 0] = cps[i].dx;
        h_cp_disp[i*3 + 1] = cps[i].dy;
        h_cp_disp[i*3 + 2] = cps[i].dz;
    }
    CUDA_CHECK(cudaMemcpy(d_CP_pos, h_cp_pos.data(), (size_t)nb * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_CP_disp, h_cp_disp.data(), (size_t)nb * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Copy mesh positions
    if (d_Mesh_pos) CUDA_CHECK(cudaFree(d_Mesh_pos));
    if (d_Mesh_disp) CUDA_CHECK(cudaFree(d_Mesh_disp));
    CUDA_CHECK(cudaMalloc(&d_Mesh_pos, (size_t)ni * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Mesh_disp, (size_t)ni * 3 * sizeof(double)));
    std::vector<double> h_mesh_pos(ni * 3);
    for (int i = 0; i < ni; ++i) {
        h_mesh_pos[i*3 + 0] = mesh[i].x;
        h_mesh_pos[i*3 + 1] = mesh[i].y;
        h_mesh_pos[i*3 + 2] = mesh[i].z;
    }
    CUDA_CHECK(cudaMemcpy(d_Mesh_pos, h_mesh_pos.data(), (size_t)ni * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate coefficient buffer (matrix_dim x 3)
    if (d_Coeffs) CUDA_CHECK(cudaFree(d_Coeffs));
    CUDA_CHECK(cudaMalloc(&d_Coeffs, (size_t)matrix_dim * 3 * sizeof(double)));
}

void RBFDeformer::deform() {
    int n = matrix_dim;
    if (n <= 0) {
        throw std::runtime_error("Invalid matrix dimension in deform().");
    }

    const size_t n_sz = (size_t)n;
    if (n_sz > std::numeric_limits<size_t>::max() / n_sz) {
        throw std::runtime_error("Matrix element count overflow in deform().");
    }
    const size_t matrix_elems = n_sz * n_sz;
    if (matrix_elems > std::numeric_limits<size_t>::max() / sizeof(double)) {
        throw std::runtime_error("Matrix byte-size overflow in deform().");
    }

    // allocate matrix and rhs
    if (!d_MatrixA) CUDA_CHECK(cudaMalloc(&d_MatrixA, matrix_elems * sizeof(double)));
    if (!d_Rhs_b) CUDA_CHECK(cudaMalloc(&d_Rhs_b, (size_t)n * 3 * sizeof(double)));

    // build full matrix on device
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    build_rbf_matrix_kernel<<<grid, block>>>(d_MatrixA, d_CP_pos, nb, supportRadiusSq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // build RHS (b): first nb rows are control point displacements, rest zeros
    // Note: cuSOLVER expects RHS to be column-major (leading dimension = n).
    // Build host RHS in column-major so columns correspond to components x,y,z.
    std::vector<double> h_b((size_t)n * 3, 0.0);
    std::vector<double> h_cp_disp(nb * 3);
    CUDA_CHECK(cudaMemcpy(h_cp_disp.data(), d_CP_disp, (size_t)nb * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nb; ++i) {
        // place component '0' (x) in column 0 contiguous, offset = 0*n + i
        h_b[0 * n + i] = h_cp_disp[i*3 + 0];
        h_b[1 * n + i] = h_cp_disp[i*3 + 1];
        h_b[2 * n + i] = h_cp_disp[i*3 + 2];
    }
    CUDA_CHECK(cudaMemcpy(d_Rhs_b, h_b.data(), (size_t)n * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // LU factorization and solve using cusolverDn
    // add a tiny regularization to the diagonal to improve conditioning
    {
        int bs = 256;
        int grid = (n + bs - 1) / bs;
        add_diag_kernel<<<grid, bs>>>(d_MatrixA, n, 1.0e-6);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, n, n, d_MatrixA, n, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER get buffer size failed (single-node deform).");
    if (!d_Work) CUDA_CHECK(cudaMalloc(&d_Work, lwork * sizeof(double)));
    if (!d_Pivots) CUDA_CHECK(cudaMalloc(&d_Pivots, n * sizeof(int)));
    if (!d_Info) CUDA_CHECK(cudaMalloc(&d_Info, sizeof(int)));

    status = cusolverDnDgetrf(cusolverH, n, n, d_MatrixA, n, d_Work, d_Pivots, d_Info);
    if (status != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER LU factorization failed (single-node deform).");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_Info, d_Info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_Info != 0) {
        char msg[128]; sprintf(msg, "Singular matrix on single-node solve. info=%d", h_Info);
        throw std::runtime_error(msg);
    }
    status = cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, n, 3, d_MatrixA, n, d_Pivots, d_Rhs_b, n, d_Info);
    if (status != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER solve failed (single-node deform).");
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy solution into d_Coeffs
    CUDA_CHECK(cudaMemcpy(d_Coeffs, d_Rhs_b, (size_t)n * 3 * sizeof(double), cudaMemcpyDeviceToDevice));

    // Keep the distributed interpolation path consistent in single-rank runs too.
    coeffs_ready = true;
    coeffs_matrix_dim = n;
    last_solve_backend = SolveBackend::SingleCuSolverDn;
}

const char* RBFDeformer::getLastSolveBackendName() const {
    switch (last_solve_backend) {
        case SolveBackend::CuSolverMpDistributedLU:
            return "cuSOLVERMp（分布式 LU）";
        case SolveBackend::RootCuSolverDnFallback:
            return "rank0 cuSOLVERDn（回退）";
        case SolveBackend::SingleCuSolverDn:
            return "cuSOLVERDn（单进程）";
        case SolveBackend::None:
        default:
            return "（未知）";
    }
}

void RBFDeformer::getResults(std::vector<MeshNode>& outNodes) {
    if (!d_Coeffs) throw std::runtime_error("Coefficients not available. Call deform() first.");
    if (!d_Mesh_pos) throw std::runtime_error("Mesh positions not initialized. Call init() first.");

    int n = matrix_dim;
    int blockSize = 256;
    int grid = (ni + blockSize - 1) / blockSize;
    interpolate_displacement_kernel<<<grid, blockSize>>>(d_Mesh_pos, d_Mesh_disp, d_CP_pos, d_Coeffs, ni, nb, supportRadiusSq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_mesh_disp((size_t)ni * 3);
    CUDA_CHECK(cudaMemcpy(h_mesh_disp.data(), d_Mesh_disp, (size_t)ni * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < ni; ++i) {
        outNodes[i].dx = h_mesh_disp[i*3 + 0];
        outNodes[i].dy = h_mesh_disp[i*3 + 1];
        outNodes[i].dz = h_mesh_disp[i*3 + 2];
    }
}

// Copy coefficients from device to host
void RBFDeformer::getCoefficientsHost(std::vector<double>& out) {
    int n = (nb > 0) ? (nb + 4) : matrix_dim;
    if (n <= 0) throw std::runtime_error("Invalid matrix dimension when getting coefficients.");
    out.resize((size_t)n * 3);
    if (!d_Coeffs) throw std::runtime_error("Device coefficients not available to copy to host.");
    CUDA_CHECK(cudaMemcpy(out.data(), d_Coeffs, (size_t)n * 3 * sizeof(double), cudaMemcpyDeviceToHost));
}

// Set coefficients on device from a host vector (also sets nb/matrix_dim)
void RBFDeformer::setCoefficientsFromHost(const std::vector<double>& in, int nb_in) {
    if (nb_in <= 0) throw std::runtime_error("Invalid nb passed to setCoefficientsFromHost.");
    nb = nb_in;
    matrix_dim = nb + 4;
    size_t expect = (size_t)matrix_dim * 3;
    if (in.size() != expect) {
        throw std::runtime_error("Coefficient host buffer size does not match expected matrix_dim*3.");
    }
    if (d_Coeffs) CUDA_CHECK(cudaFree(d_Coeffs));
    CUDA_CHECK(cudaMalloc(&d_Coeffs, expect * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Coeffs, in.data(), expect * sizeof(double), cudaMemcpyHostToDevice));
}

void RBFDeformer::setSupportRadius(double radius) {
    supportRadiusSq = radius * radius;
}

// Attempt to perform a distributed solve using cuSOLVERMp.
// This is a guarded placeholder: when compiled with -DUSE_CUSOLVERMP the
// implementation point is here; otherwise the function simply returns false
// to indicate the caller should use the centralized fallback.
    // The real implementation requires:
    //  - Initializing CAL communicator and binding to cusolverMp
    //  - Creating cusolverMp matrix descriptors for the distributed matrix A
    //    using a 2D block-cyclic distribution (proc_rows x proc_cols)
    //  - Providing pointers/IO vectors describing local blocks (row/col offsets)
    //  - Calling cusolverMpDgetrf and cusolverMpDgetrs (or similar) to factor
    //    and solve the system for each RHS column.
    //  - Ensuring the solution is available on all ranks (via cusolverMp or MPI)
    //  - Copying the solution into device memory `d_Coeffs` for local interpolation.

    // Since CAL / cusolverMp environment setup is platform-specific and
    // requires NVHPC headers/libraries, we leave the exact calls for the
    // deployment environment. Return false here so the code falls back to
    // the centralized root solve until a platform-specific implementation
    // is provided.
#ifdef USE_CUSOLVERMP
bool RBFDeformer::try_cusolvermp_solve(const std::vector<int>& owned_rows, const std::vector<double>& h_localA, int local_rows, const std::vector<double>& h_localB, int n) {
    // print local diagnostic
    std::cerr << "[cusolverMp] rank=" << mpi_rank << " local_rows=" << local_rows << " nb=" << nb << " matrix_dim=" << matrix_dim << std::endl;

    int bs = block_size > 0 ? block_size : 64;

    // Proceed with cuSOLVERMp on all ranks using fully distributed local A/B
    // (no root gather/scatter of the global matrix).
    bool ok = false;
    {
        // Create CAL communicator backed by MPI collectives (host-driven).
        CalMPIContext cal_ctx{mpi_comm};
        cal_comm_create_params_t params{};
        params.allgather = cal_mpi_allgather;
        params.req_test = cal_mpi_req_test;
        params.req_free = cal_mpi_req_free;
        params.data = &cal_ctx;
        params.nranks = mpi_size;
        params.rank = mpi_rank;
        cudaGetDevice(&params.local_device);

        cal_comm_t cal_comm = nullptr;
        calError_t cal_st = cal_comm_create(params, &cal_comm);
        std::cerr << "[debug][cusolverMp] cal_comm_create returned " << cal_st << " on rank " << mpi_rank << std::endl;
        if (cal_st != CAL_OK) {
            std::cerr << "[cusolverMp] cal_comm_create 在 rank " << mpi_rank << " 失败，code=" << cal_st << std::endl;
            return false;
        }

        cusolverMpHandle_t mpHandle = nullptr;
        cudaStream_t stream = nullptr;
        int device = 0;
        cudaGetDevice(&device);
        cudaStreamCreate(&stream);

        cusolverStatus_t cs = cusolverMpCreate(&mpHandle, device, stream);
        std::cerr << "[debug][cusolverMp] cusolverMpCreate returned " << cs << " on rank " << mpi_rank << std::endl;
        if (cs != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "[cusolverMp] cusolverMpCreate 在 rank " << mpi_rank << " 失败：" << cs << std::endl;
            cal_comm_destroy(cal_comm);
            return false;
        }

        // Create device grid.
        // For our case the RHS has only 3 columns; using a 1xP column distribution
        // can produce ranks with zero RHS columns (NUMROC == 0), which interacts
        // poorly with some collectives / allocations.
        // Use a P x 1 row distribution instead.
        int pr = mpi_size;
        int pc = 1;
        int myPr = mpi_rank;
        int myPc = 0;

        cusolverMpGrid_t grid = nullptr;
        cs = cusolverMpCreateDeviceGrid(mpHandle, &grid, cal_comm, pr, pc, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);
            std::cerr << "[debug][cusolverMp] CreateDeviceGrid returned " << cs << " on rank " << mpi_rank << std::endl;
            if (cs != CUSOLVER_STATUS_SUCCESS) {
                std::cerr << "[cusolverMp] CreateDeviceGrid 在 rank " << mpi_rank << " 失败：" << cs << std::endl;
                cusolverMpDestroy(mpHandle);
                cal_comm_destroy(cal_comm);
                return false;
            }

        // Compute local dimensions after grid creation
        int64_t local_rows_for_device = cusolverMpNUMROC(n, bs, myPr, /*isrcproc=*/0, /*nprocs=*/pr);
        int64_t local_cols_for_device = cusolverMpNUMROC(n, bs, myPc, /*isrcproc=*/0, /*nprocs=*/pc);
        int64_t local_B_cols_for_device = cusolverMpNUMROC(3, bs, myPc, /*isrcproc=*/0, /*nprocs=*/pc);

        // Create matrix descriptors for A (n x n) and B (n x 3)
        cusolverMpMatrixDescriptor_t descrA = nullptr;
        cusolverMpMatrixDescriptor_t descrB = nullptr;
        // Use local leading dimension for device A/B (column-major local storage)
        cs = cusolverMpCreateMatrixDesc(&descrA, grid, CUDA_R_64F, n, n, bs, bs, 0, 0, /*LLD*/ local_rows_for_device);
            std::cerr << "[debug][cusolverMp] CreateMatrixDesc(A) returned " << cs << " on rank " << mpi_rank << std::endl;
            if (cs != CUSOLVER_STATUS_SUCCESS) {
                std::cerr << "[cusolverMp] CreateMatrixDesc(A) 失败：" << cs << std::endl;
                cusolverMpDestroyGrid(grid); cusolverMpDestroy(mpHandle); cal_comm_destroy(cal_comm);
                return false;
            }
        // B has 3 columns globally; with pc=1 each rank owns all 3 cols and local rows
        cs = cusolverMpCreateMatrixDesc(&descrB, grid, CUDA_R_64F, n, 3, bs, bs, 0, 0, /*LLD*/ local_rows_for_device);
            std::cerr << "[debug][cusolverMp] CreateMatrixDesc(B) returned " << cs << " on rank " << mpi_rank << std::endl;
            if (cs != CUSOLVER_STATUS_SUCCESS) {
                std::cerr << "[cusolverMp] CreateMatrixDesc(B) 失败：" << cs << std::endl;
                cusolverMpDestroyMatrixDesc(descrA); cusolverMpDestroyGrid(grid); cusolverMpDestroy(mpHandle); cal_comm_destroy(cal_comm);
                return false;
            }

        // Each rank allocates local device buffers sized to its local block layout
        size_t local_block_elems = (size_t)local_rows_for_device * (size_t)local_cols_for_device;
        size_t local_B_block_elems = (size_t)local_rows_for_device * (size_t)local_B_cols_for_device;
        double* d_localAroot = nullptr;
        double* d_localBroot = nullptr;
        // Declare auxiliary variables up-front so cleanup labels don't bypass
        // initialization (required by nvcc/C++ rules when using goto).
        int64_t* d_ipiv = nullptr;
        size_t workspaceDev = 0, workspaceHost = 0;
        void* d_work = nullptr; void* h_work = nullptr;
        int info = 0;
        int* d_info = nullptr;
        size_t ws2Dev = 0, ws2Host = 0;
        void* d_work2 = nullptr; void* h_work2 = nullptr;
        int info2 = 0;
        int* d_info2 = nullptr;
        std::vector<double> fullSol;
        std::vector<double> h_localB_col;
        std::vector<double> h_localX_col;
        std::vector<double> h_localX_rows;
        int my_rows = 0;
        int total_rows = 0;
        std::vector<int> all_rows;
        std::vector<int> displs_rows;
        std::vector<int> all_owned_rows;
        std::vector<int> all_rows3;
        std::vector<int> displs_rows3;
        std::vector<double> all_X_rows;

        // Use caller-provided device-local A buffer (already block-cyclic assembled)
        bool owned_d_localAroot = true;
        if (this->d_localAroot_device != nullptr && this->d_localAroot_elems == local_block_elems && this->d_localAroot_rows == local_rows_for_device && this->d_localAroot_cols == local_cols_for_device) {
            d_localAroot = this->d_localAroot_device;
            owned_d_localAroot = false;
        } else {
            std::cerr << "[cusolverMp] rank=" << mpi_rank << " 本地 A 布局与 NUMROC 不匹配，无法全分布式求解。" << std::endl;
            goto cleanup_mp;
        }
        CUDA_CHECK(cudaMalloc(&d_localBroot, local_B_block_elems * sizeof(double)));

        // Convert local RHS from row-major [local_rows x 3] to column-major [lld x 3]
        h_localB_col.assign((size_t)local_rows_for_device * 3, 0.0);
        for (int r = 0; r < local_rows_for_device; ++r) {
            h_localB_col[(size_t)0 * local_rows_for_device + r] = h_localB[(size_t)r * 3 + 0];
            h_localB_col[(size_t)1 * local_rows_for_device + r] = h_localB[(size_t)r * 3 + 1];
            h_localB_col[(size_t)2 * local_rows_for_device + r] = h_localB[(size_t)r * 3 + 2];
        }
        CUDA_CHECK(cudaMemcpy(d_localBroot, h_localB_col.data(), local_B_block_elems * sizeof(double), cudaMemcpyHostToDevice));

        // Debug info: per-rank prints to compare expectations
        {
            std::cerr << "[cusolverMp] rank=" << mpi_rank << " grid=" << pr << "x" << pc << " coords=(" << myPr << "," << myPc << ")" << std::endl;
            std::cerr << "[cusolverMp] rank=" << mpi_rank << " owned_rows=" << local_rows << " NUMROC_rows=" << local_rows_for_device << " NUMROC_cols=" << local_cols_for_device << " NUMROC_B_cols=" << local_B_cols_for_device << std::endl;
            std::cerr << "[cusolverMp] CreateMatrixDesc 参数：MB=" << bs << " NB=" << bs << " pr=" << pr << " pc=" << pc << " LLD(A)=" << local_rows_for_device << " LLD(B)=" << local_rows_for_device << std::endl;
        }

        // Allocate pivot array on device
        CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int64_t)));

        // Workspace query
        std::cerr << "[debug][cusolverMp] Querying Getrf_bufferSize on rank " << mpi_rank << std::endl;
        cs = cusolverMpGetrf_bufferSize(mpHandle, n, n, d_localAroot, 1, 1, descrA, d_ipiv, CUDA_R_64F, &workspaceDev, &workspaceHost);
        std::cerr << "[debug][cusolverMp] Getrf_bufferSize returned " << cs << " workspaceDev=" << workspaceDev << " workspaceHost=" << workspaceHost << " on rank " << mpi_rank << std::endl;
        if (cs != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "[cusolverMp] Getrf_bufferSize 在 rank " << mpi_rank << " 失败：" << cs << std::endl;
            goto cleanup_mp;
        }
        if (workspaceDev > 0) CUDA_CHECK(cudaMalloc(&d_work, workspaceDev));
        if (workspaceHost > 0) h_work = malloc(workspaceHost);
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_info, 0, sizeof(int), stream));
        std::cerr << "[debug][cusolverMp] Calling Getrf on rank " << mpi_rank << std::endl;
        cs = cusolverMpGetrf(mpHandle, n, n, d_localAroot, 1, 1, descrA, d_ipiv, CUDA_R_64F, d_work, workspaceDev, h_work, workspaceHost, d_info);
        CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cerr << "[debug][cusolverMp] Getrf returned cs=" << cs << " info=" << info << " on rank " << mpi_rank << std::endl;
        if (cs != CUSOLVER_STATUS_SUCCESS || info != 0) {
            std::cerr << "[cusolverMp] Getrf 在 rank " << mpi_rank << " 失败：cs=" << cs << " info=" << info << std::endl;
            goto cleanup_mp;
        }
        if (mpi_rank == 0) {
            std::cerr << "[cusolverMp] Getrf 成功（cs=" << cs << "，info=" << info << "）" << std::endl;
        }

        // Solve
        std::cerr << "[debug][cusolverMp] Querying Getrs_bufferSize on rank " << mpi_rank << std::endl;
        cs = cusolverMpGetrs_bufferSize(mpHandle, CUBLAS_OP_N, n, 3, d_localAroot, 1, 1, descrA, d_ipiv, d_localBroot, 1, 1, descrB, CUDA_R_64F, &ws2Dev, &ws2Host);
        std::cerr << "[debug][cusolverMp] Getrs_bufferSize returned " << cs << " ws2Dev=" << ws2Dev << " ws2Host=" << ws2Host << " on rank " << mpi_rank << std::endl;
        if (cs != CUSOLVER_STATUS_SUCCESS) { std::cerr << "[cusolverMp] Getrs_bufferSize 在 rank " << mpi_rank << " 失败：" << cs << std::endl; goto cleanup_mp; }
        if (ws2Dev > 0) CUDA_CHECK(cudaMalloc(&d_work2, ws2Dev));
        if (ws2Host > 0) h_work2 = malloc(ws2Host);
        CUDA_CHECK(cudaMalloc(&d_info2, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_info2, 0, sizeof(int), stream));
        std::cerr << "[debug][cusolverMp] Calling Getrs on rank " << mpi_rank << std::endl;
        cs = cusolverMpGetrs(mpHandle, CUBLAS_OP_N, n, 3, d_localAroot, 1, 1, descrA, d_ipiv, d_localBroot, 1, 1, descrB, CUDA_R_64F, d_work2, ws2Dev, h_work2, ws2Host, d_info2);
        CUDA_CHECK(cudaMemcpyAsync(&info2, d_info2, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cerr << "[debug][cusolverMp] Getrs returned cs=" << cs << " info2=" << info2 << " on rank " << mpi_rank << std::endl;
        if (cs != CUSOLVER_STATUS_SUCCESS || info2 != 0) { std::cerr << "[cusolverMp] Getrs 在 rank " << mpi_rank << " 失败：cs=" << cs << " info=" << info2 << std::endl; goto cleanup_mp; }
        if (mpi_rank == 0) {
            std::cerr << "[cusolverMp] Getrs 成功（cs=" << cs << "，info=" << info2 << "）" << std::endl;
        }

        // Collect distributed solution rows to every rank (MPI_Allgatherv),
        // then reconstruct full column-major coefficients for interpolation.
        h_localX_col.assign((size_t)local_rows_for_device * 3, 0.0);
        CUDA_CHECK(cudaMemcpy(h_localX_col.data(), d_localBroot, local_B_block_elems * sizeof(double), cudaMemcpyDeviceToHost));

        h_localX_rows.assign((size_t)local_rows_for_device * 3, 0.0);
        for (int r = 0; r < local_rows_for_device; ++r) {
            h_localX_rows[(size_t)r * 3 + 0] = h_localX_col[(size_t)0 * local_rows_for_device + r];
            h_localX_rows[(size_t)r * 3 + 1] = h_localX_col[(size_t)1 * local_rows_for_device + r];
            h_localX_rows[(size_t)r * 3 + 2] = h_localX_col[(size_t)2 * local_rows_for_device + r];
        }

        my_rows = local_rows;
        all_rows.assign(mpi_size, 0);
        displs_rows.assign(mpi_size, 0);
        MPI_Allgather(&my_rows, 1, MPI_INT, all_rows.data(), 1, MPI_INT, mpi_comm);
        total_rows = 0;
        for (int r = 0; r < mpi_size; ++r) {
            displs_rows[r] = total_rows;
            total_rows += all_rows[r];
        }

        all_owned_rows.assign(total_rows, 0);
        MPI_Allgatherv(owned_rows.data(), my_rows, MPI_INT,
                       all_owned_rows.data(), all_rows.data(), displs_rows.data(), MPI_INT, mpi_comm);

        all_rows3.assign(mpi_size, 0);
        displs_rows3.assign(mpi_size, 0);
        for (int r = 0; r < mpi_size; ++r) {
            all_rows3[r] = all_rows[r] * 3;
            displs_rows3[r] = displs_rows[r] * 3;
        }
        all_X_rows.assign((size_t)total_rows * 3, 0.0);
        MPI_Allgatherv(h_localX_rows.data(), my_rows * 3, MPI_DOUBLE,
                       all_X_rows.data(), all_rows3.data(), displs_rows3.data(), MPI_DOUBLE, mpi_comm);

        fullSol.assign((size_t)n * 3, 0.0);
        for (int k = 0; k < total_rows; ++k) {
            int gi = all_owned_rows[k];
            if (gi < 0 || gi >= n) continue;
            fullSol[(size_t)0 * n + gi] = all_X_rows[(size_t)k * 3 + 0];
            fullSol[(size_t)1 * n + gi] = all_X_rows[(size_t)k * 3 + 1];
            fullSol[(size_t)2 * n + gi] = all_X_rows[(size_t)k * 3 + 2];
        }

        // Copy to device d_Coeffs for local interpolation
        if (!d_Coeffs) CUDA_CHECK(cudaMalloc(&d_Coeffs, (size_t)n * 3 * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_Coeffs, fullSol.data(), (size_t)n * 3 * sizeof(double), cudaMemcpyHostToDevice));
        ok = true;

    cleanup_mp:
        if (d_localAroot && owned_d_localAroot) cudaFree(d_localAroot);
        if (d_localBroot) cudaFree(d_localBroot);
        if (d_ipiv) cudaFree(d_ipiv);
        if (d_info) cudaFree(d_info);
        if (d_info2) cudaFree(d_info2);
        if (d_work) cudaFree(d_work);
        if (d_work2) cudaFree(d_work2);
        if (h_work) free(h_work);
        if (h_work2) free(h_work2);
        if (descrB) cusolverMpDestroyMatrixDesc(descrB);
        if (descrA) cusolverMpDestroyMatrixDesc(descrA);
        if (grid) cusolverMpDestroyGrid(grid);
        if (mpHandle) cusolverMpDestroy(mpHandle);
        if (stream) cudaStreamDestroy(stream);
        if (cal_comm) cal_comm_destroy(cal_comm);
        ;
    }

    // Ensure all ranks have the solution (root broadcast above if ok)
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, mpi_comm);
    ok = (ok_int == 1);
    return ok;
}
#else
bool RBFDeformer::try_cusolvermp_solve(const std::vector<int>& owned_rows, const std::vector<double>& h_localA, int local_rows, const std::vector<double>& h_localB, int n) {
    (void)owned_rows; (void)h_localA; (void)local_rows; (void)h_localB; (void)n;
    return false;
}
#endif
