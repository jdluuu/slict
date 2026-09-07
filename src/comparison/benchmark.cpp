#include "slict/solver_comparison.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace sc=slict::comparison;
namespace {
double RelativeError(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) {
  return (a-b).cwiseAbs().maxCoeff()/std::max(1.0,a.cwiseAbs().maxCoeff());
}
void Validate(const sc::Snapshot& s,const sc::Options& base_options) {
  const int threads=base_options.threads;
  const auto a=sc::LinearizeNative(s,s.initial,threads);
  const Eigen::MatrixXd H=a.jacobian.transpose()*a.jacobian;
  const Eigen::VectorXd g=a.jacobian.transpose()*a.residual;
  for (auto backend:{sc::Backend::CeresScalar,sc::Backend::CeresBatch,sc::Backend::NativeBatch}) {
    const auto other=backend==sc::Backend::NativeBatch ? sc::LinearizeNativeBatch(s,s.initial,threads) : sc::LinearizeCeres(s,s.initial,backend,threads);
    const double h=RelativeError(H,other.jacobian.transpose()*other.jacobian);
    const double gradient=RelativeError(g,other.jacobian.transpose()*other.residual);
    const double cost=std::abs(a.residual.squaredNorm()-other.residual.squaredNorm())/std::max(1.0,a.residual.squaredNorm());
    if (std::max({h,gradient,cost})>1e-9) throw std::runtime_error("A/B/C linearization mismatch: "+std::string(sc::BackendName(backend)));
    std::cout<<"validated frame="<<s.frame<<" backend="<<sc::BackendName(backend)<<" H="<<h<<" gradient="<<gradient<<" cost="<<cost<<'\n';
  }
  for (int iterations : {1, 8}) {
    sc::Options options=base_options;
    options.threads = threads;
    options.iterations = iterations;
    options.backend = sc::Backend::CeresScalar;
    const auto b = sc::Solve(s, options);
    options.backend = sc::Backend::CeresBatch;
    const auto c = sc::Solve(s, options);
    const double state_error = sc::StateDistance(b.state, c.state);
    if (!b.metrics.usable || !c.metrics.usable || state_error > 1e-6)
      throw std::runtime_error("B/C solved state mismatch at frame " + std::to_string(s.frame));
    std::cout << "validated state frame=" << s.frame << " iterations=" << iterations
              << " B_C_max_state_error=" << state_error << '\n';
    options.backend=sc::Backend::Native;
    const auto native=sc::Solve(s,options);
    options.backend=sc::Backend::NativeBatch;
    const auto batch=sc::Solve(s,options);
    const double native_error=sc::StateDistance(native.state,batch.state);
    const double native_cost_error=std::abs(native.metrics.final_cost-batch.metrics.final_cost)/
        std::max(1.0,std::abs(native.metrics.final_cost));
    if (!native.metrics.usable || !batch.metrics.usable || native_error>1e-6 || native_cost_error>1e-7)
      throw std::runtime_error("Native/batch solved state or cost mismatch at frame "+std::to_string(s.frame));
    std::cout<<"validated native state frame="<<s.frame<<" iterations="<<iterations
             <<" A_D_max_state_error="<<native_error<<" cost="<<native_cost_error<<'\n';
  }
}
void Help() {
  std::cout<<"SLICT frozen-problem A/B/C benchmark (no ROS master required)\n"
    <<"  --snapshots DIR|FILE       Read .slict snapshots\n"
    <<"  --synthetic N             Generate N synthetic windows (default 3)\n"
    <<"  --generate DIR            Save generated snapshots; no timing without --output\n"
    <<"  --knots N --lidar N --imu N  Synthetic dimensions (default 20/1000/60)\n"
    <<"  --backend native|ceres_scalar|ceres_batch|native_batch|all (default all)\n"
    <<"  --threads N --iterations N --repeat N --warmup N (default 1/1/5/1)\n"
    <<"  --ceres-initial-radius R   Common B/C initial trust region radius (default 1e4)\n"
    <<"  --validate N              Validate first N windows before timing (default 1)\n"
    <<"  --max-snapshots N          Limit number of loaded windows\n"
    <<"  --output FILE             CSV output (default build/abc.csv)\n";
}
int Number(const std::map<std::string,std::string>& args,const std::string& key,int fallback,int minimum=0) {
  const auto it=args.find(key);if (it==args.end()) return fallback;
  std::size_t pos=0;const int value=std::stoi(it->second,&pos);
  if (pos!=it->second.size() || value<minimum) throw std::invalid_argument("Invalid "+key);
  return value;
}
}
int main(int argc,char** argv) {
  try {
    std::map<std::string,std::string> args;
    const std::vector<std::string> names={"--snapshots","--synthetic","--generate","--knots","--lidar","--imu","--backend","--threads","--iterations","--repeat","--warmup","--validate","--max-snapshots","--output","--ceres-initial-radius"};
    for (int i=1;i<argc;++i) {
      const std::string key=argv[i];if (key=="--help") {Help();return 0;}
      if (std::find(names.begin(),names.end(),key)==names.end() || i+1==argc) throw std::invalid_argument("Unknown/incomplete option: "+key);
      if (!args.emplace(key,argv[++i]).second) throw std::invalid_argument("Duplicate option: "+key);
    }
    Eigen::setNbThreads(1);
    std::vector<sc::Snapshot> snapshots;
    const int maximum=Number(args,"--max-snapshots",100000,1);
    if (args.count("--snapshots")) {
      if (args.count("--synthetic") || args.count("--generate")) throw std::invalid_argument("Choose recorded or synthetic input");
      const std::filesystem::path path=args.at("--snapshots");std::vector<std::filesystem::path> files;
      if (std::filesystem::is_regular_file(path)) files.push_back(path);
      else for (const auto& entry:std::filesystem::directory_iterator(path))
        if (entry.is_regular_file() && entry.path().extension()==".slict") files.push_back(entry.path());
      std::sort(files.begin(),files.end());if (files.size()>static_cast<std::size_t>(maximum)) files.resize(maximum);
      for (const auto& file:files) {auto s=sc::LoadSnapshot(file.string());s.Canonicalize();snapshots.push_back(std::move(s));}
    } else {
      const int count=std::min(maximum,Number(args,"--synthetic",3,1));
      for (int i=0;i<count;++i) snapshots.push_back(sc::MakeSyntheticSnapshot(Number(args,"--knots",20,4),Number(args,"--lidar",1000),Number(args,"--imu",60),42+i));
      if (args.count("--generate")) {
        std::filesystem::create_directories(args.at("--generate"));
        for (std::size_t i=0;i<snapshots.size();++i) sc::SaveSnapshot(snapshots[i],(std::filesystem::path(args.at("--generate"))/("synthetic_"+std::to_string(i)+".slict")).string());
        std::cout<<"Saved "<<snapshots.size()<<" synthetic windows to "<<args.at("--generate")<<'\n';
        if (!args.count("--output")) return 0;
      }
    }
    if (snapshots.empty()) throw std::runtime_error("No snapshots found");
    sc::Options options;options.threads=Number(args,"--threads",1,1);options.iterations=Number(args,"--iterations",1,1);
    if (args.count("--ceres-initial-radius")) {
      std::size_t pos=0;
      const auto& value=args.at("--ceres-initial-radius");
      options.ceres_initial_trust_region_radius=std::stod(value,&pos);
      if (pos!=value.size() || !std::isfinite(options.ceres_initial_trust_region_radius) ||
          options.ceres_initial_trust_region_radius<=0 || options.ceres_initial_trust_region_radius>1e16)
        throw std::invalid_argument("Ceres initial radius must be in (0, 1e16]");
    }
    const int repeats=Number(args,"--repeat",5,1),warmups=Number(args,"--warmup",1);
    std::vector<sc::Backend> backends={sc::Backend::Native,sc::Backend::CeresScalar,sc::Backend::CeresBatch,sc::Backend::NativeBatch};
    if (args.count("--backend") && args.at("--backend")!="all") backends={sc::ParseBackend(args.at("--backend"))};
    for (int i=0;i<std::min<int>(snapshots.size(),Number(args,"--validate",1));++i) Validate(snapshots[i],options);
    std::vector<std::string> digests;for (const auto& s:snapshots) digests.push_back(sc::SnapshotDigest(s));
    const std::string output=args.count("--output")?args.at("--output"):"build/abc.csv";
    const auto parent=std::filesystem::path(output).parent_path();if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream csv(output);if (!csv) throw std::runtime_error("Cannot open CSV: "+output);sc::WriteCsvHeader(csv);
    std::map<sc::Backend,std::vector<double>> timings;
    int failures=0;
    for (int repeat=-warmups;repeat<repeats;++repeat) {
      for (std::size_t i=0;i<snapshots.size();++i) {
        for (std::size_t b=0;b<backends.size();++b) {
          // Rotate A/B/C order between windows/repetitions. Single-backend mode
          // supports isolated process RSS and external ABCCBA scheduling.
          options.backend=backends[(b+i+repeat+warmups)%backends.size()];
          auto result=sc::Solve(snapshots[i],options);
          if (repeat>=0) {
            sc::WriteCsvRow(csv,snapshots[i],options,result.metrics,repeat,digests[i]);
            timings[options.backend].push_back(result.metrics.total_ms);
            if (!result.metrics.usable) ++failures;
          }
        }
      }
    }
    csv.close();if (!csv) throw std::runtime_error("Failed writing CSV");
    for (auto& entry:timings) {
      auto& times=entry.second;std::sort(times.begin(),times.end());
      std::cout<<sc::BackendName(entry.first)<<" n="<<times.size()<<" mean_ms="<<std::accumulate(times.begin(),times.end(),0.)/times.size()
        <<" p95_ms="<<times[static_cast<std::size_t>(std::ceil(.95*times.size()))-1]<<'\n';
    }
    std::cout<<"CSV: "<<output<<"; unusable="<<failures<<". Compare final_cost as well as time; A and Ceres use different step strategies.\n";
    return failures?2:0;
  } catch (const std::exception& e) {std::cerr<<"benchmark: "<<e.what()<<'\n';return 1;}
}
