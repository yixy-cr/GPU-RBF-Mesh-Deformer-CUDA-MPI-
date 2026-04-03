#include "MeshLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>

bool MeshLoader::loadMsh(const std::string& filepath) {
    last_msh_timing_ = FileTiming{};
    const auto t_total_0 = std::chrono::steady_clock::now();

    const auto t_open_0 = std::chrono::steady_clock::now();
    std::ifstream file(filepath);
    const auto t_open_1 = std::chrono::steady_clock::now();
    last_msh_timing_.open_s = std::chrono::duration<double>(t_open_1 - t_open_0).count();
    if (!file.is_open()) {
        std::cerr << "Error: Could not open mesh file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    bool in_nodes_section = false;
    long num_nodes = 0;
    long nodes_read = 0;

    while (true) {
        const auto t_read_0 = std::chrono::steady_clock::now();
        if (!std::getline(file, line)) break;
        const auto t_read_1 = std::chrono::steady_clock::now();
        last_msh_timing_.read_s += std::chrono::duration<double>(t_read_1 - t_read_0).count();
        last_msh_timing_.bytes_read += (long long)line.size() + 1;
        last_msh_timing_.lines_total += 1;

        if (line.find("$Nodes") == 0) {
            in_nodes_section = true;
            // The next line should contain the number of nodes
            {
                const auto t_hdr_0 = std::chrono::steady_clock::now();
                const auto t_hdr_read_0 = std::chrono::steady_clock::now();
                if (!std::getline(file, line)) return false;
                const auto t_hdr_read_1 = std::chrono::steady_clock::now();
                last_msh_timing_.read_s += std::chrono::duration<double>(t_hdr_read_1 - t_hdr_read_0).count();
                last_msh_timing_.bytes_read += (long long)line.size() + 1;
                last_msh_timing_.lines_total += 1;
                try {
                    num_nodes = std::stol(line);
                    meshNodes.reserve(num_nodes);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Invalid number of nodes format in " << filepath << std::endl;
                    return false;
                }
                const auto t_hdr_1 = std::chrono::steady_clock::now();
                last_msh_timing_.header_s += std::chrono::duration<double>(t_hdr_1 - t_hdr_0).count();
            }
            continue;
        }

        if (line.find("$EndNodes") == 0) {
            in_nodes_section = false;
            break;
        }

        if (in_nodes_section) {
            // If we encounter another section marker (starting with '$'),
            // assume the nodes section ended (some writers omit $EndNodes).
            if (!line.empty() && line[0] == '$') {
                in_nodes_section = false;
                break;
            }
            const auto t_parse_0 = std::chrono::steady_clock::now();
            std::stringstream ss(line);
            MeshNode node;
            // Assumes format: <node-id> <x> <y> <z>
            if (ss >> node.id >> node.x >> node.y >> node.z) {
                node.dx = 0.0;
                node.dy = 0.0;
                node.dz = 0.0;
                meshNodes.push_back(node);
                nodes_read++;
            }
            const auto t_parse_1 = std::chrono::steady_clock::now();
            last_msh_timing_.parse_s += std::chrono::duration<double>(t_parse_1 - t_parse_0).count();
        }
    }

    const auto t_total_1 = std::chrono::steady_clock::now();
    last_msh_timing_.total_s = std::chrono::duration<double>(t_total_1 - t_total_0).count();

    if (nodes_read != num_nodes) {
        std::cerr << "Warning: Read " << nodes_read << " nodes, but expected " << num_nodes << "." << std::endl;
    }

    std::cout << "Successfully loaded " << meshNodes.size() << " mesh nodes from " << filepath << "." << std::endl;
    return true;
}

bool MeshLoader::isDuplicate(const ControlPoint& point, double tolerance) {
    double tol_sq = tolerance * tolerance;
    for (const auto& existing_point : controlPoints) {
        double dx = point.x - existing_point.x;
        double dy = point.y - existing_point.y;
        double dz = point.z - existing_point.z;
        if ((dx * dx + dy * dy + dz * dz) < tol_sq) {
            return true;
        }
    }
    return false;
}

bool MeshLoader::loadSidesetPoints(const std::string& filepath, int sample_step) {
    last_sideset_timing_ = FileTiming{};
    const auto t_total_0 = std::chrono::steady_clock::now();
    const auto t_open_0 = std::chrono::steady_clock::now();
    std::ifstream file(filepath);
    const auto t_open_1 = std::chrono::steady_clock::now();
    last_sideset_timing_.open_s = std::chrono::duration<double>(t_open_1 - t_open_0).count();
    if (!file.is_open()) {
        std::cerr << "Error: Could not open sideset file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    int points_added = 0;
    int points_skipped = 0;

    // Skip the header line if it exists
    {
        const auto t_hdr_0 = std::chrono::steady_clock::now();
        const auto t_read_0 = std::chrono::steady_clock::now();
        std::getline(file, line);
        const auto t_read_1 = std::chrono::steady_clock::now();
        last_sideset_timing_.read_s += std::chrono::duration<double>(t_read_1 - t_read_0).count();
        last_sideset_timing_.bytes_read += (long long)line.size() + 1;
        last_sideset_timing_.lines_total += 1;
        const auto t_hdr_1 = std::chrono::steady_clock::now();
        last_sideset_timing_.header_s += std::chrono::duration<double>(t_hdr_1 - t_hdr_0).count();
    }

    int line_idx = 0; // index of data lines (0-based)
    while (true) {
        const auto t_read_0 = std::chrono::steady_clock::now();
        if (!std::getline(file, line)) break;
        const auto t_read_1 = std::chrono::steady_clock::now();
        last_sideset_timing_.read_s += std::chrono::duration<double>(t_read_1 - t_read_0).count();
        last_sideset_timing_.bytes_read += (long long)line.size() + 1;
        last_sideset_timing_.lines_total += 1;

        // Apply downsampling if requested
        if (sample_step > 1) {
            if ((line_idx % sample_step) != 0) {
                line_idx++;
                continue;
            }
            line_idx++;
        }

        last_sideset_timing_.lines_sampled += 1;

        const auto t_parse_0 = std::chrono::steady_clock::now();
        std::stringstream ss(line);
        std::string value;
        
        ControlPoint cp;
        int idM;

        // Format: idM, x, y, z, p, s00
        try {
            std::getline(ss, value, ','); idM = std::stoi(value);
            std::getline(ss, value, ','); cp.x = std::stod(value);
            std::getline(ss, value, ','); cp.y = std::stod(value);
            std::getline(ss, value, ','); cp.z = std::stod(value);
            std::getline(ss, value, ','); cp.p = std::stod(value);
            std::getline(ss, value, ','); cp.s00 = std::stod(value);
        } catch (const std::invalid_argument& e) {
            std::cerr << "警告：无法解析该行：" << line << std::endl;
            last_sideset_timing_.parse_errors += 1;
            continue;
        }
        (void)idM;

        const auto t_parse_1 = std::chrono::steady_clock::now();
        last_sideset_timing_.parse_s += std::chrono::duration<double>(t_parse_1 - t_parse_0).count();


        // Initialize displacements to zero
        cp.dx = 0.0;
        cp.dy = 0.0;
        cp.dz = 0.0;

        {
            const auto t_dedup_0 = std::chrono::steady_clock::now();
            if (!isDuplicate(cp)) {
                controlPoints.push_back(cp);
                points_added++;
            } else {
                points_skipped++;
            }
            const auto t_dedup_1 = std::chrono::steady_clock::now();
            last_sideset_timing_.dedup_s += std::chrono::duration<double>(t_dedup_1 - t_dedup_0).count();
        }
    }

    const auto t_total_1 = std::chrono::steady_clock::now();
    last_sideset_timing_.total_s = std::chrono::duration<double>(t_total_1 - t_total_0).count();
    last_sideset_timing_.points_added = points_added;
    last_sideset_timing_.points_skipped = points_skipped;

    std::cout << "已成功从 " << filepath << " 读取 sideset 点。" << std::endl;
    std::cout << "  - 新增：" << points_added << " 个控制点（sample_step=" << sample_step << "）。" << std::endl;
    if(points_skipped > 0) {
        std::cout << "  - 跳过：" << points_skipped << " 个重复点。" << std::endl;
    }
    return true;
}
