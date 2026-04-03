#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <cstdint>

#include "common.h"
#include "MeshLoader.h"
#include "RBFDeformer.h"
#include <mpi.h>
#include <cstdlib>

// Note: This file will be extended to perform distributed execution using
// MPI + CUDA + cuSOLVERMp. The current change adds MPI init/finalize and
// broadcasts control points so subsequent patches can implement distributed
// solving and interpolation.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to save the final mesh with displacements to a CSV file
void save_results_csv(const std::string& filename, const std::vector<MeshNode>& nodes) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "错误：无法打开输出文件：" << filename << std::endl;
        return;
    }

    // Set precision for output
    outfile << std::fixed << std::setprecision(8);

    // Write header
    outfile << "node_id,x,y,z,dx,dy,dz" << std::endl;

    // Write node data
    for (const auto& node : nodes) {
        outfile << node.id << ","
                << node.x << "," << node.y << "," << node.z << ","
                << node.dx << "," << node.dy << "," << node.dz << std::endl;
    }
    std::cout << "结果已保存到 " << filename << std::endl;
}

void save_results_csv_local(const std::string& filename, const std::vector<MeshNode>& nodes, int start_idx, int count) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "错误：无法打开输出文件：" << filename << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(8);
    outfile << "node_id,x,y,z,dx,dy,dz" << std::endl;

    for (int i = 0; i < count; ++i) {
        const auto& node = nodes[start_idx + i];
        outfile << node.id << ","
                << node.x << "," << node.y << "," << node.z << ","
                << node.dx << "," << node.dy << "," << node.dz << std::endl;
    }
    std::cout << "本地结果已保存到 " << filename
              << "（start=" << start_idx << ", count=" << count << "）" << std::endl;
}

// Save a deformed Gmsh .msh by replacing the $Nodes section in the original mesh
void save_deformed_msh(const std::string& original_msh, const std::string& out_msh, const std::vector<MeshNode>& nodes) {
    std::ifstream fin(original_msh);
    if (!fin.is_open()) {
        std::cerr << "错误：无法打开原始 msh 进行读取：" << original_msh << std::endl;
        return;
    }
    std::ofstream fout(out_msh);
    if (!fout.is_open()) {
        std::cerr << "错误：无法打开输出 msh 进行写入：" << out_msh << std::endl;
        return;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.rfind("$Nodes", 0) == 0) {
            // Write $Nodes and replace node block with our deformed nodes
            fout << line << "\n";
            // consume original count line
            std::string count_line;
            if (!std::getline(fin, count_line)) break;
            // write new count
            fout << nodes.size() << "\n";
            // write nodes: id x y z (apply displacement)
            for (const auto& n : nodes) {
                double x = n.x + n.dx;
                double y = n.y + n.dy;
                double z = n.z + n.dz;
                fout << n.id << " " << std::setprecision(12) << x << " " << y << " " << z << "\n";
            }

            // skip original node lines until we hit a line starting with '$'
            std::string next_marker;
            while (std::getline(fin, next_marker)) {
                if (!next_marker.empty() && next_marker[0] == '$') {
                    // if it's $EndNodes, consume and continue
                    if (next_marker == "$EndNodes") {
                        // write $EndNodes then continue copying rest
                        fout << "$EndNodes\n";
                    } else {
                        // write $EndNodes to be safe, then write this marker as next section
                        fout << "$EndNodes\n";
                        fout << next_marker << "\n";
                    }
                    break;
                }
            }
            // Now copy the rest of the file
            while (std::getline(fin, line)) {
                fout << line << "\n";
            }
            break; // done
        } else {
            fout << line << "\n";
        }
    }

    fin.close();
    fout.close();
    std::cout << "已写出变形后的 msh：" << out_msh << std::endl;
}


int main() {
    std::cout << "--- 使用 CUDA 的 RBF 网格变形 ---" << std::endl;

    // --- Initialize MPI ---
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Make multi-rank logging less confusing: reduce stdout/stderr buffering.
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    // Select CUDA device based on rank (simple round-robin)
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (world_size > 1 && device_count > 0 && world_size > device_count) {
        const bool allow_oversub = (std::getenv("RBF_ALLOW_GPU_OVERSUBSCRIBE") != nullptr);
        if (!allow_oversub) {
            if (world_rank == 0) {
                const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
                std::cerr << "错误：MPI 进程数(" << world_size << ") > 可见 CUDA 设备数(" << device_count
                          << ")。当前 CUDA_VISIBLE_DEVICES=" << (cvd ? cvd : "(null)")
                          << "。这会导致多 rank 复用同一 GPU，默认禁止。"
                          << " 如需强制运行请设置 RBF_ALLOW_GPU_OVERSUBSCRIBE=1。" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 3);
        } else if (world_rank == 0) {
            std::cerr << "警告：已设置 RBF_ALLOW_GPU_OVERSUBSCRIBE=1，允许多 rank 复用同一 GPU。"
                      << " 该模式可能显著降低性能并增大通信失败概率。" << std::endl;
        }
    }
    if (device_count > 0) {
        int device_id = world_rank % device_count;
        cudaSetDevice(device_id);
        if (world_rank == 0) std::cout << "Using " << device_count << " CUDA devices; rank 0 sets device " << device_id << std::endl;
        // Print per-rank GPU assignment and properties for clearer logging
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
            std::cerr << "[rank " << world_rank << "] CUDA_VISIBLE_DEVICES=" << (cvd?cvd:"(null)")
                      << ", assigned_device_id=" << device_id
                      << ", name=\"" << prop.name << "\""
                      << ", pciBusID=" << prop.pciBusID
                      << ", pciDeviceID=" << prop.pciDeviceID
                      << ", totalMem=" << prop.totalGlobalMem
                      << std::endl;
        } else {
            std::cerr << "[rank " << world_rank << "] Failed to get device properties for device " << device_id << std::endl;
        }
    }

    // --- 1. Configuration ---
    const std::string msh_filepath = "data/1pinflue10mm_gmsh.msh";
    const double support_radius = 0.01; // increased support radius to better match target displacement
    const double rotation_angle_deg = 5.0;

    const bool timing_barrier = []() {
        if (const char* e = std::getenv("RBF_TIMING_BARRIER")) {
            return std::atoi(e) != 0;
        }
        return true;
    }();

    auto reduce_and_print_time = [&](const char* label, double local_s) {
        double sum_s = 0.0;
        double min_s = 0.0;
        double max_s = 0.0;
        MPI_Reduce(&local_s, &sum_s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_s, &min_s, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (world_rank == 0) {
            const double avg_s = sum_s / (double)world_size;
            std::cout << "[Timing] " << label
                      << ": min/avg/max = "
                      << (min_s * 1e3) << "/" << (avg_s * 1e3) << "/" << (max_s * 1e3)
                      << " ms" << std::endl;
        }
    };

    auto reduce_and_print_ll = [&](const char* label, long long local_v) {
        long long sum_v = 0;
        long long min_v = 0;
        long long max_v = 0;
        MPI_Reduce(&local_v, &sum_v, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_v, &min_v, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_v, &max_v, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        if (world_rank == 0) {
            const double avg_v = (double)sum_v / (double)world_size;
            std::cout << "[Count]  " << label
                      << ": min/avg/max = "
                      << min_v << "/" << avg_v << "/" << max_v
                      << " (sum=" << sum_v << ")" << std::endl;
        }
    };

    // --- 2. Load Data ---
    MeshLoader loader;
    std::cout << "\nLoading mesh data..." << std::endl;
    // Fully distributed mode: all ranks load the same mesh file locally,
    // then each rank computes/interpolates only its own deterministic slice.
    const double mesh_t0 = MPI_Wtime();
    if (!loader.loadMsh(msh_filepath)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    const double mesh_t1 = MPI_Wtime();
    reduce_and_print_time("mesh.loadMsh total", mesh_t1 - mesh_t0);
    const auto& tm = loader.getLastMshTiming();
    if (world_rank == 0) {
        std::cout << "[Timing] mesh.loadMsh breakdown(ms): open=" << (tm.open_s * 1e3)
                  << ", header=" << (tm.header_s * 1e3)
                  << ", read=" << (tm.read_s * 1e3)
                  << ", parse=" << (tm.parse_s * 1e3)
                  << ", total=" << (tm.total_s * 1e3)
                  << ", bytes=" << tm.bytes_read
                  << ", lines=" << tm.lines_total
                  << std::endl;
    }

    // --- 2b. Load control points (distributed) ---
    // Each rank reads a subset of sideset files, then rank 0 merges + de-duplicates
    // and broadcasts the final control-point list to all ranks.
    int num_sidesets = 16;
    int sample_N = 100;
    int target_cp = 0;
    if (const char* e = std::getenv("RBF_NUM_SIDESETS")) {
        int v = std::atoi(e);
        if (v > 0) num_sidesets = v;
    }
    if (const char* e = std::getenv("RBF_SAMPLE_N")) {
        int v = std::atoi(e);
        if (v > 0) sample_N = v;
    }
    if (const char* e = std::getenv("RBF_TARGET_CP")) {
        int v = std::atoi(e);
        if (v > 0) target_cp = v;
    }
    if (world_rank == 0) {
        std::cout << "\nLoading control points from " << num_sidesets
                  << " sideset files with downsampling (N=" << sample_N << ")..." << std::endl;
        if (target_cp > 0) {
            std::cout << "[INFO] RBF_TARGET_CP=" << target_cp
                      << "（将在全局去重后按均匀索引精确下采样）" << std::endl;
        }
    }

    loader.controlPoints.clear();

    MeshLoader::FileTiming sideset_acc;
    double sideset_loop_total_s = 0.0;
    {
        char buf[256];
        const double t0 = MPI_Wtime();
        for (int si = world_rank; si < num_sidesets; si += world_size) {
            snprintf(buf, sizeof(buf), "data/txt/sideset4_exe_%d.txt", si);
            const std::string sideset_filepath(buf);
            std::cerr << "[rank " << world_rank << "] reading sideset " << si
                      << ": " << sideset_filepath << " (sample_step=" << sample_N << ")" << std::endl;
            if (!loader.loadSidesetPoints(sideset_filepath, sample_N)) {
                std::cerr << "[rank " << world_rank << "] Warning: failed to load " << sideset_filepath << "." << std::endl;
            } else {
                const auto& ts = loader.getLastSidesetTiming();
                sideset_acc.open_s += ts.open_s;
                sideset_acc.header_s += ts.header_s;
                sideset_acc.read_s += ts.read_s;
                sideset_acc.parse_s += ts.parse_s;
                sideset_acc.dedup_s += ts.dedup_s;
                sideset_acc.total_s += ts.total_s;
                sideset_acc.bytes_read += ts.bytes_read;
                sideset_acc.lines_total += ts.lines_total;
                sideset_acc.lines_sampled += ts.lines_sampled;
                sideset_acc.parse_errors += ts.parse_errors;
                sideset_acc.points_added += ts.points_added;
                sideset_acc.points_skipped += ts.points_skipped;
            }
        }
        const double t1 = MPI_Wtime();
        sideset_loop_total_s = (t1 - t0);
    }

    if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "\n[Timing] Sideset file I/O+parse (per-rank stats; units=ms unless stated)" << std::endl;
    }
    reduce_and_print_time("sideset.loop_total", sideset_loop_total_s);
    reduce_and_print_time("sideset.open_sum", sideset_acc.open_s);
    reduce_and_print_time("sideset.header_sum", sideset_acc.header_s);
    reduce_and_print_time("sideset.read_sum", sideset_acc.read_s);
    reduce_and_print_time("sideset.parse_sum", sideset_acc.parse_s);
    reduce_and_print_time("sideset.dedup_sum", sideset_acc.dedup_s);
    reduce_and_print_time("sideset.total_sum", sideset_acc.total_s);
    reduce_and_print_ll("sideset.bytes_read", sideset_acc.bytes_read);
    reduce_and_print_ll("sideset.lines_total", sideset_acc.lines_total);
    reduce_and_print_ll("sideset.lines_sampled", sideset_acc.lines_sampled);
    reduce_and_print_ll("sideset.parse_errors", sideset_acc.parse_errors);
    reduce_and_print_ll("sideset.points_added", sideset_acc.points_added);
    reduce_and_print_ll("sideset.points_skipped", sideset_acc.points_skipped);

    // --- Parallel control-point merge (no rank-0 bottleneck) ---
    // 1) Local de-dup on each rank
    struct Key { int64_t x, y, z; };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            size_t h = 1469598103934665603ull;
            auto mix = [&](uint64_t v) {
                h ^= v;
                h *= 1099511628211ull;
            };
            mix((uint64_t)k.x);
            mix((uint64_t)k.y);
            mix((uint64_t)k.z);
            return h;
        }
    };
    struct KeyEq {
        bool operator()(const Key& a, const Key& b) const noexcept {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }
    };
    constexpr double dup_tol = 1e-6; // match MeshLoader::isDuplicate default tolerance
    auto make_key = [&](double x, double y, double z) -> Key {
        return Key{(int64_t)llround(x / dup_tol), (int64_t)llround(y / dup_tol), (int64_t)llround(z / dup_tol)};
    };
    auto dedup_inplace = [&](std::vector<ControlPoint>& cps) {
        std::unordered_map<Key, int, KeyHash, KeyEq> seen;
        seen.reserve(cps.size());
        std::vector<ControlPoint> out;
        out.reserve(cps.size());
        for (const auto& cp : cps) {
            Key k = make_key(cp.x, cp.y, cp.z);
            if (seen.find(k) == seen.end()) {
                seen.emplace(k, (int)out.size());
                out.push_back(cp);
            }
        }
        cps.swap(out);
    };

    const int local_before = (int)loader.controlPoints.size();

    double t_stageB_local_dedup_s = 0.0;
    double t_stageB_allgather_s = 0.0;
    double t_stageB_pack_s = 0.0;
    double t_stageB_global_dedup_s = 0.0;
    double t_stageB_sort_s = 0.0;

    if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
    {
        const double t0 = MPI_Wtime();
    dedup_inplace(loader.controlPoints);
        const double t1 = MPI_Wtime();
        t_stageB_local_dedup_s = (t1 - t0);
    }
    const int local_nb = (int)loader.controlPoints.size();
    if (local_before != local_nb) {
        std::cerr << "[rank " << world_rank << "] local control points dedup: "
                  << local_before << " -> " << local_nb << std::endl;
    }

    // 2) Global exchange: all ranks exchange their local unique CP list in parallel
    std::vector<int> all_nb(world_size, 0);
    if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
    {
        const double t0 = MPI_Wtime();
        MPI_Allgather(&local_nb, 1, MPI_INT, all_nb.data(), 1, MPI_INT, MPI_COMM_WORLD);
        const double t1 = MPI_Wtime();
        t_stageB_allgather_s += (t1 - t0);
    }

    int total_doubles = 0;
    int total_points_exchange = 0;
    {
        const double t0 = MPI_Wtime();
        std::vector<double> local_buf((size_t)local_nb * 8);
    for (int i = 0; i < local_nb; ++i) {
        const ControlPoint& cp = loader.controlPoints[i];
        local_buf[i*8 + 0] = cp.x;
        local_buf[i*8 + 1] = cp.y;
        local_buf[i*8 + 2] = cp.z;
        local_buf[i*8 + 3] = cp.dx;
        local_buf[i*8 + 4] = cp.dy;
        local_buf[i*8 + 5] = cp.dz;
        local_buf[i*8 + 6] = cp.p;
        local_buf[i*8 + 7] = cp.s00;
    }

    std::vector<int> recvcounts_d(world_size), displs_d(world_size);
    total_doubles = 0;
    for (int r = 0; r < world_size; ++r) {
        recvcounts_d[r] = all_nb[r] * 8;
        displs_d[r] = total_doubles;
        total_doubles += recvcounts_d[r];
    }
    total_points_exchange = total_doubles / 8;

    std::vector<double> gathered_buf((size_t)total_doubles);
        const double t1 = MPI_Wtime();
        t_stageB_pack_s = (t1 - t0);

        if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
        const double t2 = MPI_Wtime();
        MPI_Allgatherv(local_buf.data(), local_nb * 8, MPI_DOUBLE,
                       gathered_buf.data(), recvcounts_d.data(), displs_d.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        const double t3 = MPI_Wtime();
        t_stageB_allgather_s += (t3 - t2);

        // 3) Global de-dup: each rank de-dups the full set locally (results identical)
        const double t4 = MPI_Wtime();
        std::unordered_map<Key, int, KeyHash, KeyEq> seen_global;
        seen_global.reserve((size_t)total_points_exchange);

        std::vector<ControlPoint> merged;
        merged.reserve((size_t)total_points_exchange);
        for (int i = 0; i < total_points_exchange; ++i) {
            ControlPoint cp;
            cp.x = gathered_buf[i*8 + 0];
            cp.y = gathered_buf[i*8 + 1];
            cp.z = gathered_buf[i*8 + 2];
            cp.dx = gathered_buf[i*8 + 3];
            cp.dy = gathered_buf[i*8 + 4];
            cp.dz = gathered_buf[i*8 + 5];
            cp.p = gathered_buf[i*8 + 6];
            cp.s00 = gathered_buf[i*8 + 7];
            Key k = make_key(cp.x, cp.y, cp.z);
            if (seen_global.find(k) == seen_global.end()) {
                seen_global.emplace(k, (int)merged.size());
                merged.push_back(cp);
            }
        }
        const double t5 = MPI_Wtime();
        t_stageB_global_dedup_s = (t5 - t4);

        // Deterministic order across ranks (helps reproducibility)
        const double t6 = MPI_Wtime();
        std::sort(merged.begin(), merged.end(), [&](const ControlPoint& a, const ControlPoint& b) {
            Key ka = make_key(a.x, a.y, a.z);
            Key kb = make_key(b.x, b.y, b.z);
            if (ka.x != kb.x) return ka.x < kb.x;
            if (ka.y != kb.y) return ka.y < kb.y;
            return ka.z < kb.z;
        });
        const double t7 = MPI_Wtime();
        t_stageB_sort_s = (t7 - t6);

        loader.controlPoints.swap(merged);
    }
    int nb = (int)loader.controlPoints.size();

    // Optional: exact-cap control-point count for limit testing.
    // Applied after global merge+dedup+sort to keep deterministic behavior across ranks.
    if (target_cp > 0 && nb > target_cp) {
        std::vector<ControlPoint> capped;
        capped.reserve((size_t)target_cp);
        const size_t src_n = loader.controlPoints.size();
        const size_t dst_n = (size_t)target_cp;
        for (size_t i = 0; i < dst_n; ++i) {
            size_t idx = (i * src_n) / dst_n;
            if (idx >= src_n) idx = src_n - 1;
            capped.push_back(loader.controlPoints[idx]);
        }
        loader.controlPoints.swap(capped);
        nb = (int)loader.controlPoints.size();
        if (world_rank == 0) {
            std::cout << "[INFO] RBF_TARGET_CP 生效：全局控制点数裁剪为 " << nb << std::endl;
        }
    } else if (target_cp > 0 && world_rank == 0) {
        std::cout << "[INFO] RBF_TARGET_CP=" << target_cp
                  << " >= 当前控制点数 " << nb
                  << "，不执行裁剪。" << std::endl;
    }

    if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "\n[Timing] StageB(control points merge/exchange) (per-rank stats)" << std::endl;
    }
    reduce_and_print_time("stageB.local_dedup", t_stageB_local_dedup_s);
    reduce_and_print_time("stageB.pack_buf", t_stageB_pack_s);
    reduce_and_print_time("stageB.allgather+allgatherv", t_stageB_allgather_s);
    reduce_and_print_time("stageB.global_dedup", t_stageB_global_dedup_s);
    reduce_and_print_time("stageB.sort", t_stageB_sort_s);
    reduce_and_print_time("stageB.total", (t_stageB_local_dedup_s + t_stageB_pack_s + t_stageB_allgather_s + t_stageB_global_dedup_s + t_stageB_sort_s));

    if (world_rank == 0) {
        int global_before = 0;
        for (int r = 0; r < world_size; ++r) global_before += all_nb[r];
        std::cout << "合并后控制点数：" << nb
                  << "（本地去重后全局条目数 " << global_before
                  << "，全局交换后条目数 " << total_points_exchange << "）" << std::endl;
    }
    if (nb == 0) {
        if (world_rank == 0) {
            std::cerr << "致命错误：未读取到任何控制点，终止运行。" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Capture MPI library version for reproducibility logging
    char mpi_ver_buf[MPI_MAX_LIBRARY_VERSION_STRING];
    int mpi_ver_len = 0;
    std::string mpi_library_version;
    if (MPI_Get_library_version(mpi_ver_buf, &mpi_ver_len) == MPI_SUCCESS && mpi_ver_len > 0) {
        mpi_library_version.assign(mpi_ver_buf, mpi_ver_buf + mpi_ver_len);
        // trim trailing newlines
        while (!mpi_library_version.empty() && (mpi_library_version.back() == '\n' || mpi_library_version.back() == '\r')) {
            mpi_library_version.pop_back();
        }
    }

    // start timer
    double t_start = MPI_Wtime();

    // --- 3. Define Boundary Conditions (Cantilever Bending Test) ---
    std::cout << "\n计算悬臂梁弯曲位移..." << std::endl;
    // Physical scenario: bottom (z=0) fixed, top (z=height) pushed horizontally by max_displacement
    double height = 0.01; // 10 mm
    double max_displacement = 0.002; // 2 mm

    if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
    const double t_disp_0 = MPI_Wtime();

    for (auto& cp : loader.controlPoints) {
        double ratio = cp.z / height;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;

        // Parabolic bending (more pronounced near top)
        cp.dx = max_displacement * ratio * ratio;
        cp.dy = 0.0;
        cp.dz = 0.0;
    }
    if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
    const double t_disp_1 = MPI_Wtime();
    reduce_and_print_time("disp.compute_control_point_displacements", (t_disp_1 - t_disp_0));
    std::cout << "已计算 " << loader.controlPoints.size() << " 个控制点的位移。" << std::endl;

    // --- 4. Run RBF Deformation on GPU ---
    try {
        RBFDeformer deformer;
        deformer.init_distributed(MPI_COMM_WORLD);

        // For logging: capture merged CP count before adding anchors.
        // (On rank 0, `nb` is the merged+deduped control-point count at this point.)
        int nb_merged_before_anchors = nb;

        // --- Anchor bottom nodes to prevent rigid-body leakage ---
        // Add a small set of control points at z ~= 0 with zero displacement
        const int max_anchors = 200;
        const double z_tol = 1e-6;
        std::vector<int> bottom_idx;
        for (size_t i = 0; i < loader.meshNodes.size(); ++i) {
            if (std::abs(loader.meshNodes[i].z) <= z_tol) bottom_idx.push_back((int)i);
        }
        if (!bottom_idx.empty()) {
            int step = std::max(1, (int)bottom_idx.size() / max_anchors);
            int anchors_added = 0;
            auto is_dup = [&](const ControlPoint& cp)->bool{
                for (const auto &e : loader.controlPoints) {
                    double dx = cp.x - e.x;
                    double dy = cp.y - e.y;
                    double dz = cp.z - e.z;
                    if (dx*dx + dy*dy + dz*dz < 1e-12) return true;
                }
                return false;
            };
            for (size_t k = 0; k < bottom_idx.size(); k += step) {
                const auto &n = loader.meshNodes[bottom_idx[k]];
                ControlPoint a;
                a.x = n.x; a.y = n.y; a.z = n.z;
                a.dx = 0.0; a.dy = 0.0; a.dz = 0.0;
                a.p = 0.0; a.s00 = 0.0;
                if (!is_dup(a)) {
                    loader.controlPoints.push_back(a);
                    anchors_added++;
                }
            }
            std::cout << "已添加 " << anchors_added << " 个底部锚点控制点。" << std::endl;
        }

        // In fully distributed mode every rank already has identical mesh/control-point
        // sets after stage-B allgather+sort, and applies the same deterministic anchor logic.
        nb = (int)loader.controlPoints.size();

        // Ensure Y-displacements of control points are zero on all ranks
        for (size_t i = 0; i < loader.controlPoints.size(); ++i) {
            loader.controlPoints[i].dy = 0.0;
        }

        if (world_size == 1) {
            std::cout << "\n在 GPU 上初始化 RBF 变形器..." << std::endl;
            deformer.init(loader.meshNodes, loader.controlPoints, support_radius);
            std::cout << "\n执行变形求解..." << std::endl;
            deformer.deform();
            std::cout << "\n从 GPU 取回结果..." << std::endl;
            deformer.getResults(loader.meshNodes);
        } else {
            // Distributed solve + parallel interpolation using cuSOLVERMp/NCCL
            int ni_total = 0;

            // Initialize distributed state in deformer
            deformer.init_distributed(MPI_COMM_WORLD);

            // All ranks have control points (was broadcast earlier). Call distributed solve.
            if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
            const double t_solve_0 = MPI_Wtime();
            deformer.solve_distributed(loader.controlPoints, MPI_COMM_WORLD);
            if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
            const double t_solve_1 = MPI_Wtime();
            reduce_and_print_time("solve.distributed", (t_solve_1 - t_solve_0));

            // Deterministic local partition on every rank (no root scatter/gather)
            if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
            const double t_interp_0 = MPI_Wtime();
            ni_total = (int)loader.meshNodes.size();

            // compute per-rank node counts
            std::vector<int> counts(world_size, 0);
            int base = ni_total / world_size;
            int rem = ni_total % world_size;
            for (int r = 0; r < world_size; ++r) counts[r] = base + (r < rem ? 1 : 0);

            int local_n = counts[world_rank];
            int local_start = 0;
            for (int r = 0; r < world_rank; ++r) local_start += counts[r];

            // Build local mesh nodes and interpolate locally
            std::vector<MeshNode> local_mesh(local_n);
            for (int i = 0; i < local_n; ++i) {
                const auto& src = loader.meshNodes[local_start + i];
                local_mesh[i].id = src.id;
                local_mesh[i].x = src.x;
                local_mesh[i].y = src.y;
                local_mesh[i].z = src.z;
                local_mesh[i].dx = local_mesh[i].dy = local_mesh[i].dz = 0.0;
            }

            // Each rank performs interpolation on its subset using device coeffs
            deformer.interpolate_distributed(local_mesh, loader.controlPoints, MPI_COMM_WORLD);

            // Write local displacements back to each rank's local partition view
            std::vector<double> local_disp((size_t)local_n * 3);
            for (int i = 0; i < local_n; ++i) {
                local_disp[i*3 + 0] = local_mesh[i].dx;
                local_disp[i*3 + 1] = local_mesh[i].dy;
                local_disp[i*3 + 2] = local_mesh[i].dz;
                loader.meshNodes[local_start + i].dx = local_mesh[i].dx;
                loader.meshNodes[local_start + i].dy = local_mesh[i].dy;
                loader.meshNodes[local_start + i].dz = local_mesh[i].dz;
            }

            if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
            const double t_interp_1 = MPI_Wtime();
            reduce_and_print_time("interp.distributed_total", (t_interp_1 - t_interp_0));
        }

        // --- 5. Save Results (merge to single files on rank 0) ---
        if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
        const double t_out_0 = MPI_Wtime();
        if (world_size == 1) {
            if (world_rank == 0) {
                std::cout << "\n保存结果...（rank 0）" << std::endl;
                save_results_csv("bending4.csv", loader.meshNodes);
                save_deformed_msh(msh_filepath, "bending4.msh", loader.meshNodes);
            }
        } else {
            int ni_total = (int)loader.meshNodes.size();
            int base = ni_total / world_size;
            int rem = ni_total % world_size;
            int local_n = base + (world_rank < rem ? 1 : 0);
            int local_start = 0;
            for (int r = 0; r < world_rank; ++r) {
                local_start += base + (r < rem ? 1 : 0);
            }

            // Gather local displacement partitions to rank 0, then write single merged outputs.
            std::vector<double> local_disp3((size_t)local_n * 3);
            for (int i = 0; i < local_n; ++i) {
                const auto& node = loader.meshNodes[local_start + i];
                local_disp3[(size_t)i * 3 + 0] = node.dx;
                local_disp3[(size_t)i * 3 + 1] = node.dy;
                local_disp3[(size_t)i * 3 + 2] = node.dz;
            }

            std::vector<int> recv_counts3;
            std::vector<int> recv_displs3;
            std::vector<double> all_disp3;
            if (world_rank == 0) {
                recv_counts3.resize(world_size);
                recv_displs3.resize(world_size);
                int disp = 0;
                for (int r = 0; r < world_size; ++r) {
                    int cnt_nodes = base + (r < rem ? 1 : 0);
                    recv_counts3[r] = cnt_nodes * 3;
                    recv_displs3[r] = disp;
                    disp += recv_counts3[r];
                }
                all_disp3.resize((size_t)ni_total * 3);
            }

            MPI_Gatherv(local_disp3.data(), local_n * 3, MPI_DOUBLE,
                        world_rank == 0 ? all_disp3.data() : nullptr,
                        world_rank == 0 ? recv_counts3.data() : nullptr,
                        world_rank == 0 ? recv_displs3.data() : nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (world_rank == 0) {
                for (int i = 0; i < ni_total; ++i) {
                    loader.meshNodes[i].dx = all_disp3[(size_t)i * 3 + 0];
                    loader.meshNodes[i].dy = all_disp3[(size_t)i * 3 + 1];
                    loader.meshNodes[i].dz = all_disp3[(size_t)i * 3 + 2];
                }
                std::cout << "\n保存合并结果...（rank 0）" << std::endl;
                save_results_csv("bending4.csv", loader.meshNodes);
                save_deformed_msh(msh_filepath, "bending4.msh", loader.meshNodes);
                std::cout << "提示：多进程模式已在运行期聚合为单一输出：bending4.csv 与 bending4.msh。" << std::endl;
            }
        }
        if (timing_barrier) MPI_Barrier(MPI_COMM_WORLD);
        const double t_out_1 = MPI_Wtime();
        reduce_and_print_time("output.write_files", (t_out_1 - t_out_0));


        // --- Run summary (rank 0 only) ---
        if (world_rank == 0) {
            const char* require_mp = std::getenv("RBF_REQUIRE_CUSOLVERMP");
            const char* disable_mp = std::getenv("RBF_DISABLE_CUSOLVERMP");
            const char* ompi_ucc = std::getenv("OMPI_MCA_coll_ucc_enable");
            const char* ompi_hcoll = std::getenv("OMPI_MCA_coll_hcoll_enable");

            std::cout << "\n=== 运行摘要 ===" << std::endl;
            std::cout << "MPI 进程数：" << world_size << std::endl;
            if (!mpi_library_version.empty()) {
                std::cout << "MPI 库版本：" << mpi_library_version << std::endl;
            }
            std::cout << "控制点输入：sideset 文件数=" << num_sidesets
                      << "，下采样步长(N)=" << sample_N
                      << "，合并去重后控制点数=" << nb_merged_before_anchors
                      << "，加锚点后控制点数=" << (int)loader.controlPoints.size()
                      << std::endl;
            std::cout << "计算模式：读取=" << (world_size > 1 ? "分布式(各rank本地加载)" : "单进程")
                      << "，求解=" << deformer.getLastSolveBackendName()
                      << "，插值=" << (world_size > 1 ? "分布式(本地分片)" : "单进程")
                      << std::endl;
            std::cout << "环境变量：RBF_REQUIRE_CUSOLVERMP=" << (require_mp ? require_mp : "(未设置)")
                      << "，RBF_DISABLE_CUSOLVERMP=" << (disable_mp ? disable_mp : "(未设置)")
                      << "，OMPI_MCA_coll_ucc_enable=" << (ompi_ucc ? ompi_ucc : "(未设置)")
                      << "，OMPI_MCA_coll_hcoll_enable=" << (ompi_hcoll ? ompi_hcoll : "(未设置)")
                      << std::endl;
            std::cout << "复现实验命令（示例）：" << std::endl;
            std::cout << "  source ./env_nvhpc_24.9.sh && "
                      << "RBF_NUM_SIDESETS=" << num_sidesets << " RBF_SAMPLE_N=" << sample_N << " "
                      << "RBF_REQUIRE_CUSOLVERMP=" << (require_mp ? require_mp : "1") << " "
                      << "mpirun -np " << world_size << " --bind-to none ./rbf_deformer" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "\n--- 发生异常 ---" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- 变形流程已成功完成 ---" << std::endl;
    double t_end = MPI_Wtime();
    if (world_rank == 0) {
        std::cout << "总运行时间：" << (t_end - t_start) << " 秒" << std::endl;
    }

    // Ensure all ranks have completed GPU/MPI work before finalizing MPI.
    // This helps avoid rare finalize-time crashes when async CUDA or MPI
    // progress threads are still tearing down.
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
