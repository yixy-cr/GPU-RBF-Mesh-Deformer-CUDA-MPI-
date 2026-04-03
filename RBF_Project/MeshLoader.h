#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <vector>
#include <string>
#include "common.h"

class MeshLoader {
public:
    struct FileTiming {
        double open_s = 0.0;
        double header_s = 0.0;
        double read_s = 0.0;
        double parse_s = 0.0;
        double dedup_s = 0.0;
        double total_s = 0.0;
        long long bytes_read = 0;
        long long lines_total = 0;
        long long lines_sampled = 0;
        long long parse_errors = 0;
        long long points_added = 0;
        long long points_skipped = 0;
    };

    std::vector<MeshNode> meshNodes;
    std::vector<ControlPoint> controlPoints;

    /**
     * @brief Loads mesh nodes from a Gmsh .msh file (Version 2 ASCII format).
     * @param filepath Path to the .msh file.
     * @return True if loading was successful, false otherwise.
     * @note This parser is simplified and expects the node format:
     *       <node-id> <x> <y> <z>
     */
    bool loadMsh(const std::string& filepath);

    /**
     * @brief Loads control points from a sideset .txt file with optional downsampling.
     *
     * The file format is expected to be "idM, x, y, z, p, s00" per line.
     * This function includes logic to prevent adding duplicate points
     * based on a spatial tolerance of 1e-6.
     *
     * @param filepath Path to the sideset file.
     * @param sample_step If >1, only every `sample_step`-th data line is kept (downsampling).
     * @return True if loading was successful, false otherwise.
     */
    bool loadSidesetPoints(const std::string& filepath, int sample_step = 1);

    const FileTiming& getLastMshTiming() const { return last_msh_timing_; }
    const FileTiming& getLastSidesetTiming() const { return last_sideset_timing_; }

private:
    /**
     * @brief Checks if a control point is already present in the list.
     * @param point The point to check.
     * @param tolerance The distance tolerance for the duplication check.
     * @return True if the point is a duplicate, false otherwise.
     */
    bool isDuplicate(const ControlPoint& point, double tolerance = 1e-6);

    FileTiming last_msh_timing_;
    FileTiming last_sideset_timing_;
};

#endif // MESH_LOADER_H
