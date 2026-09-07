#include "../tmnSolver.h"
#include "internal.h"
#include <filesystem>
#include <iomanip>
#include <sstream>

bool tmnSolver::SolveComparison(PoseSplineX &traj, Vector3d &BIG, Vector3d &BIA,
                               map<int,int> &curr_knot_x, int swNextBase, int iter,
                               deque<deque<ImuSequence>> &SwImuBundle,
                               deque<vector<LidarCoef>> &SwLidarCoef,
                               vector<ImuIdx> &imuSelected, vector<lidarFeaIdx> &featureSelected,
                               string &description, slict::OptStat &report, slict::TimeLog &tlog)
{
    namespace sc = slict::comparison;
    const auto input_start = sc::Clock::now();
    if (iter == max_outer_iters - 1) {
        ++comparison_frame;
        comparison_bg_reference = BIG; comparison_ba_reference = BIA;
    }
    sc::Snapshot snapshot;
    snapshot.frame = comparison_frame; snapshot.outer_iteration = iter;
    snapshot.start_time = traj.minTime(); snapshot.dt = traj.getDt();
    snapshot.native_damping = lambda / std::pow(2.0, (max_outer_iters - 1) - iter);
    snapshot.initial.gyro_bias = BIG; snapshot.initial.accel_bias = BIA;
    snapshot.gyro_reference = comparison_bg_reference; snapshot.accel_reference = comparison_ba_reference;
    snapshot.gravity = GRAV; snapshot.imu_weights << GYR_N, ACC_N, GYR_W, ACC_W;
    for (int i=0; i<traj.numKnots(); ++i) {
        snapshot.initial.rotations.push_back(traj.getKnotSO3(i));
        snapshot.initial.positions.push_back(traj.getKnotPos(i));
    }
    if (comparison_fuse_imu) for (const auto &index : imuSelected) {
        const auto &measurement = SwImuBundle[index.i][index.j][index.k];
        const auto support = traj.computeTIndex(measurement.t);
        sc::ImuObservation o; o.span = support.second; o.u = support.first;
        o.timestamp = measurement.t; o.gyro = measurement.gyro; o.accel = measurement.acce;
        snapshot.imu.push_back(o);
    }
    if (comparison_fuse_lidar) for (const auto &index : featureSelected) {
        const auto &coef = SwLidarCoef[index.wdidx][index.pointidx];
        const auto support = traj.computeTIndex(coef.t);
        sc::LidarObservation o; o.span = support.second; o.u = support.first; o.timestamp = coef.t;
        o.point = coef.f; o.normal = coef.n.head<3>(); o.offset = coef.n[3]; o.weight = coef.plnrty * lidar_weight;
        snapshot.lidar.push_back(o);
    }
    const int current_base = curr_knot_x.begin()->first;
    if (fuse_marg && !comparison_prior.Empty()) {
        snapshot.prior = comparison_prior;
        for (int &knot : snapshot.prior.knots) knot += comparison_prior_base - current_base;
    } else {
        // All three backends inherit this initial anchor through marginalization.
        snapshot.fixed_knots = {0};
    }
    snapshot.Canonicalize(); snapshot.Validate();
    const double input_ms = sc::Milliseconds(input_start);
    const std::string digest = sc::SnapshotDigest(snapshot);
    if (!comparison_snapshot_dir.empty() && iter == max_outer_iters - 1 &&
        (comparison_frame - 1) % comparison_snapshot_stride == 0 && comparison_saved < comparison_snapshot_limit) {
        std::ostringstream name; name << "window_" << std::setfill('0') << std::setw(6) << comparison_frame << ".slict";
        sc::SaveSnapshot(snapshot, (std::filesystem::path(comparison_snapshot_dir)/name.str()).string());
        ++comparison_saved;
    }
    const auto result = sc::Solve(snapshot, comparison_options);
    if (!result.metrics.usable)
        throw std::runtime_error("Comparison backend failed; refusing a silent fallback to a different objective: " + result.metrics.termination);
    const auto update_start = sc::Clock::now();
    for (int i=0; i<traj.numKnots(); ++i)
        traj.setKnot(SE3d(result.state.rotations[i], result.state.positions[i]), i);
    BIG = result.state.gyro_bias; BIA = result.state.accel_bias;
    const double writeback_ms = sc::Milliseconds(update_start);
    double marginalize_ms = 0;
    if (iter == 0 && fuse_marg) {
        const auto marginalize_start = sc::Clock::now();
        comparison_prior = sc::Marginalize(snapshot, result.state, swNextBase-current_base, comparison_options.threads);
        comparison_prior_base = swNextBase;
        marginalize_ms = sc::Milliseconds(marginalize_start);
    }
    if (comparison_csv.is_open()) {
        sc::WriteCsvRow(comparison_csv, snapshot, comparison_options, result.metrics, 0, digest, marginalize_ms, input_ms+writeback_ms);
        comparison_csv.flush();
        if (!comparison_csv) throw std::runtime_error("Writing comparison CSV failed");
    }
    report.surfFactors = snapshot.lidar.size(); report.imuFactors = snapshot.imu.size();
    report.propFactors = !snapshot.prior.Empty(); report.velFactors = 0;
    report.J0 = 2*result.metrics.initial_cost; report.JK = 2*result.metrics.final_cost;
    report.J0Surf = report.JKSurf = report.J0Imu = report.JKImu = -1;
    report.J0Prop = report.JKProp = report.J0Vel = report.JKVel = -1;
    report.tbuildceres = result.metrics.build_ms; report.tslv = result.metrics.solve_ms;
    tlog.t_prep.push_back(result.metrics.build_ms+input_ms);
    tlog.t_hprior.push_back(0); tlog.t_himu.push_back(0); tlog.t_hlidar.push_back(0);
    tlog.t_compute.push_back(result.metrics.solve_ms);
    tlog.t_update.push_back(writeback_ms); tlog.t_marg.push_back(marginalize_ms);
    tlog.t_solve.push_back(input_ms+result.metrics.total_ms+writeback_ms+marginalize_ms);
    tlog.ceres_iter = result.metrics.iterations;
    std::ostringstream message;
    message << comparison_backend_name << ": build " << result.metrics.build_ms << " ms, solve "
            << result.metrics.solve_ms << " ms, total " << result.metrics.total_ms << " ms, marg "
            << marginalize_ms << " ms, blocks " << result.metrics.residual_blocks;
    description = message.str();
    // The frontend changes associations between single LM steps. Preserve the
    // radius selected by Ceres instead of restarting its damping at every call.
    // Update after writing this call's CSV, which records the input radius.
    if (sc::UsesCeresSolver(comparison_options.backend) && result.metrics.ceres_final_trust_region_radius > 0)
        comparison_options.ceres_initial_trust_region_radius = result.metrics.ceres_final_trust_region_radius;
    return true;
}
