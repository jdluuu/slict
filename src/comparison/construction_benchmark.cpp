#include "internal.h"
#include <splbatch/construction_diagnostics.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace sc = slict::comparison;
#ifdef SPLBATCH_CONSTRUCTION_DIAGNOSTICS
namespace cd = splbatch::construction_diagnostics;
#endif

namespace {
int Number(const std::map<std::string,std::string>& args, const std::string& name,
           int fallback, int minimum) {
  auto it=args.find(name); if (it==args.end()) return fallback;
  std::size_t end=0; const int value=std::stoi(it->second,&end);
  if (end!=it->second.size() || value<minimum) throw std::invalid_argument("Invalid "+name);
  return value;
}
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
#ifdef SPLBATCH_CONSTRUCTION_DIAGNOSTICS
struct Calibration { double leaf_ns=0, child_gap_ns=0, leaf_wall_ns=0; };
Calibration Calibrate() {
  constexpr int count=100000;
  std::vector<double> leaf, gap, wall;
  for (int round=0;round<7;++round) {
    cd::Profile profile;
    cd::Session session(&profile);
    const auto start=sc::Clock::now();
    for (int i=0;i<count;++i) { cd::Scope probe(cd::Stage::Calibration,false); }
    wall.push_back(sc::Milliseconds(start)*1e6/count);
    const auto& c=profile.stages[static_cast<std::size_t>(cd::Stage::Calibration)];
    leaf.push_back(c.exclusive_ns/count);
    for (int i=0;i<count;++i) {
      cd::Scope parent(cd::Stage::ObservationStorage,false);
      cd::Scope child(cd::Stage::Prepare,false);
    }
    const auto& parent=profile.stages[static_cast<std::size_t>(cd::Stage::ObservationStorage)];
    gap.push_back(parent.exclusive_ns/count-leaf.back());
  }
  auto median=[](auto values){std::sort(values.begin(),values.end());return values[values.size()/2];};
  return {median(leaf),median(gap),median(wall)};
}
void CheckProfile(const sc::Snapshot& s, const cd::Profile& p, int blocks) {
  Check(p.observations==s.imu.size()+s.lidar.size(),"Diagnostic observation count changed");
  Check(p.stages[static_cast<std::size_t>(cd::Stage::Binding)].calls==p.sampled_observations,
        "Diagnostic binding selection mismatch");
  Check(p.stages[static_cast<std::size_t>(cd::Stage::Prepare)].calls==p.sampled_observations,
        "Diagnostic preparation selection mismatch");
  Check(p.stages[static_cast<std::size_t>(cd::Stage::BatchCreation)].calls==
        static_cast<unsigned>(blocks-!s.prior.Empty()),"Diagnostic batch count mismatch");
  double raw=0;
  for (const auto& c:p.stages) { Check(c.exclusive_ns>=0,"Nested diagnostic scopes overlap"); raw+=c.exclusive_ns; }
  Check(raw>=0,"Invalid profile");
}
#endif
}  // namespace

int main(int argc,char** argv) {
  try {
    std::map<std::string,std::string> args;
    const std::vector<std::string> names={"--snapshots","--output","--threads","--repeat","--warmup",
      "--stride","--offset","--enabled","--validate","--max-snapshots"};
    for (int i=1;i<argc;++i) {
      const std::string key=argv[i];
      if (key=="--help") {
        std::cout<<"Construction attribution, same native_batch single-step solve.\n"
          <<"--snapshots DIR --output CSV [--threads 4 --repeat 1 --warmup 1]\n"
          <<"[--stride 16 --offset 0 --enabled 1 --validate 0 --max-snapshots 101]\n"
          <<"Control executable has no construction probes. Diagnostic stride samples observation scopes only.\n";
        return 0;
      }
      if (std::find(names.begin(),names.end(),key)==names.end() || i+1==argc)
        throw std::invalid_argument("Unknown/incomplete option: "+key);
      if (!args.emplace(key,argv[++i]).second) throw std::invalid_argument("Duplicate option: "+key);
    }
    Check(args.count("--snapshots") && args.count("--output"),"--snapshots and --output are required");
    const int repeat=Number(args,"--repeat",1,1),warmup=Number(args,"--warmup",1,0);
    const int stride=Number(args,"--stride",16,1),offset=Number(args,"--offset",0,0);
    const bool enabled=Number(args,"--enabled",1,0)!=0;
    Check(offset<stride && stride<=4096,"Invalid sampling stride/offset");
    const int maximum=Number(args,"--max-snapshots",101,1);
    const int validate=Number(args,"--validate",0,0);
    Eigen::setNbThreads(1);
    std::vector<std::filesystem::path> files;
    const std::filesystem::path input=args.at("--snapshots");
    if (std::filesystem::is_regular_file(input)) files.push_back(input);
    else for (const auto& item:std::filesystem::directory_iterator(input))
      if (item.is_regular_file() && item.path().extension()==".slict") files.push_back(item.path());
    std::sort(files.begin(),files.end());
    if (files.size()>static_cast<std::size_t>(maximum)) files.resize(maximum);
    Check(!files.empty(),"No snapshots found");
    std::vector<sc::Snapshot> snapshots;
    std::vector<std::string> digests;
    for (const auto& file:files) {
      auto snapshot=sc::LoadSnapshot(file.string()); snapshot.Canonicalize();
      digests.push_back(sc::SnapshotDigest(snapshot)); snapshots.push_back(std::move(snapshot));
    }
    sc::Options options;options.backend=sc::Backend::NativeBatch;
    options.threads=Number(args,"--threads",4,1);options.iterations=1;
    double max_j=0,max_r=0,max_state=0;
    for (int i=0;i<std::min<int>(validate,snapshots.size());++i) {
      const auto& s=snapshots[i];
      sc::Linearization batch;
      sc::Result solved;
#ifdef SPLBATCH_CONSTRUCTION_DIAGNOSTICS
      cd::Profile profile; profile.stride=1;
      {
        cd::Session session(&profile);
        batch=sc::LinearizeNativeBatch(s,s.initial,options.threads);
      }
      profile={};
      {
        cd::Session session(&profile);
        solved=sc::Solve(s,options);
      }
      CheckProfile(s,profile,solved.metrics.residual_blocks);
      double sum=0;for (const auto& c:profile.stages) sum+=c.exclusive_ns;
      Check(sum*1e-6<=solved.metrics.build_ms,"Profile exceeds measured build interval");
#else
      batch=sc::LinearizeNativeBatch(s,s.initial,options.threads);
      solved=sc::Solve(s,options);
#endif
      const auto native=sc::LinearizeNative(s,s.initial,options.threads);
      max_j=std::max(max_j,(batch.jacobian-native.jacobian).cwiseAbs().maxCoeff()/
                    std::max(1.,native.jacobian.cwiseAbs().maxCoeff()));
      max_r=std::max(max_r,(batch.residual-native.residual).cwiseAbs().maxCoeff()/
                    std::max(1.,native.residual.cwiseAbs().maxCoeff()));
      auto reference_options=options;reference_options.backend=sc::Backend::Native;
      const auto reference=sc::Solve(s,reference_options);
      max_state=std::max(max_state,sc::StateDistance(reference.state,solved.state));
      Check(solved.metrics.usable && reference.metrics.usable,"Validation solve failed");
    }
    Check(max_j<1e-9 && max_r<1e-9 && max_state<1e-6,"Instrumented result differs from native reference");
    std::cout<<"validation_windows="<<std::min<int>(validate,snapshots.size())
             <<" max_relative_J="<<max_j<<" max_relative_r="<<max_r<<" max_state_error="<<max_state<<'\n';
    const std::filesystem::path output=args.at("--output");
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    std::ofstream csv(output);Check(bool(csv),"Cannot open output");csv<<std::setprecision(17);
    csv<<"snapshot,frame,repeat,mode,threads,stride,offset,observations,residual_blocks,build_ms,total_ms,destroy_ms,final_cost,usable";
#ifdef SPLBATCH_CONSTRUCTION_DIAGNOSTICS
    const auto calibration=Calibrate();
    csv<<",sampled_observations,empty_leaf_ns,empty_child_gap_ns,empty_leaf_wall_ns";
    for (const char* name:cd::Names)
      csv<<','<<name<<"_raw_ns,"<<name<<"_calls,"<<name<<"_estimated_ns,"<<name<<"_estimated_calls,"<<name<<"_estimated_children";
#else
    (void)enabled;
#endif
    csv<<'\n';
    std::vector<double> build_times;
    for (int r=-warmup;r<repeat;++r) for (std::size_t i=0;i<snapshots.size();++i) {
      sc::Result result;
      const int selected_offset=(offset+std::max(0,r))%stride;
#ifdef SPLBATCH_CONSTRUCTION_DIAGNOSTICS
      cd::Profile profile;profile.stride=stride;profile.offset=selected_offset;
      {
        cd::Session session(enabled?&profile:nullptr);
        result=sc::Solve(snapshots[i],options);
      }
      if (enabled) CheckProfile(snapshots[i],profile,result.metrics.residual_blocks);
      const char* mode=enabled?"diagnostic":"diagnostic_disabled";
#else
      result=sc::Solve(snapshots[i],options);
      const char* mode="control";
#endif
      Check(result.metrics.usable,"Unusable solve");
      if (r<0) continue;
      const auto& m=result.metrics;
      csv<<digests[i]<<','<<snapshots[i].frame<<','<<r<<','<<mode<<','<<options.threads<<','<<stride<<','<<selected_offset
         <<','<<snapshots[i].imu.size()+snapshots[i].lidar.size()<<','<<m.residual_blocks
         <<','<<m.build_ms<<','<<m.total_ms<<','<<m.destroy_ms<<','<<m.final_cost<<','<<m.usable;
#ifdef SPLBATCH_CONSTRUCTION_DIAGNOSTICS
      csv<<','<<profile.sampled_observations<<','<<calibration.leaf_ns<<','<<calibration.child_gap_ns<<','<<calibration.leaf_wall_ns;
      for (const auto& c:profile.stages)
        csv<<','<<c.exclusive_ns<<','<<c.calls<<','<<c.weighted_ns<<','<<c.weighted_calls<<','<<c.weighted_children;
#endif
      csv<<'\n';build_times.push_back(m.build_ms);
    }
    csv.close();Check(bool(csv),"Writing CSV failed");
    std::cout<<"rows="<<build_times.size()<<" mean_build_ms="
             <<std::accumulate(build_times.begin(),build_times.end(),0.)/build_times.size()<<'\n';
    return 0;
  } catch (const std::exception& e) {std::cerr<<"construction benchmark: "<<e.what()<<'\n';return 1;}
}
