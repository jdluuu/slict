#include "internal.h"
#include "evaluators.h"
#include <splbatch/problem_batcher.hpp>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

namespace sc=slict::comparison;
namespace {
void Check(bool ok,const std::string& message) {if (!ok) throw std::runtime_error(message);}
void Near(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b,double tolerance,const std::string& label) {
  Check(a.rows()==b.rows() && a.cols()==b.cols(),label+" shape");
  Check(a.allFinite() && b.allFinite(),label+" nonfinite");
  if (a.size()==0) return;
  const double error=(a-b).cwiseAbs().maxCoeff()/std::max(1.,a.cwiseAbs().maxCoeff());
  Check(error<tolerance,label+" relative error="+std::to_string(error));
}
void AddPrior(sc::Snapshot& s) {
  s.prior.knots={0,2,3};
  for (int i:s.prior.knots) {
    s.prior.reference.rotations.push_back(s.initial.rotations[i]*sc::Rotation::exp(sc::Vector3(.02,-.01,.03)));
    s.prior.reference.positions.push_back(s.initial.positions[i]+sc::Vector3(.01,-.03,.02));
  }
  s.prior.reference.gyro_bias=s.initial.gyro_bias; s.prior.reference.accel_bias=s.initial.accel_bias;
  s.prior.sqrt_information=Eigen::MatrixXd::Identity(24,24);
  for (int r=0;r<24;++r) for (int c=0;c<24;++c) s.prior.sqrt_information(r,c)+=.03*std::sin(r+3*c+1.);
  s.prior.offset=Eigen::VectorXd::LinSpaced(24,-.03,.03);
}
void FactorTest(bool imu,int situation,sc::Backend backend) {
  auto s=sc::MakeSyntheticSnapshot(4,imu?0:1,imu?1:0,54);s.fixed_knots.clear();
  const double u=situation==0?0.0:situation==1?1.0:.4321;
  if (imu) s.imu[0].u=u;else s.lidar[0].u=u;
  for (int i=0;i<4;++i) {
    if (situation==0) s.initial.rotations[i]=sc::Rotation();
    if (situation==1) s.initial.rotations[i]=sc::Rotation::exp(sc::Vector3(1e-8*i,-2e-8*i,3e-8*i));
    if (situation==2) s.initial.rotations[i]=sc::Rotation::exp(sc::Vector3(.2+.13*i,-.3+.05*i,.4-.06*i));
  }
  const auto native=imu?sc::NativeImuFactor(s,s.initial,s.imu[0]):sc::NativeLidarFactor(s,s.initial,s.lidar[0]);
  auto state=s.initial;auto bundle=sc::BuildCeresProblem(s,state,backend);
  std::vector<ceres::ResidualBlockId> ids;bundle.problem->GetResidualBlocks(&ids);Check(ids.size()==1,"Single factor count");
  auto* cost=bundle.problem->GetCostFunctionForResidualBlock(ids[0]);
  std::vector<double*> parameters;bundle.problem->GetParameterBlocksForResidualBlock(ids[0],&parameters);
  const int rows=cost->num_residuals();const auto sizes=cost->parameter_block_sizes();
  for (int mode=0;mode<5;++mode) {
    std::vector<std::vector<double>> buffers(sizes.size());std::vector<double*> jacobians(sizes.size(),nullptr);
    for (std::size_t b=0;b<sizes.size();++b) {
      buffers[b].assign(rows*sizes[b],std::numeric_limits<double>::quiet_NaN());
      if (mode==0 || (mode==1 && b%2) || (mode==2 && b<4)) jacobians[b]=buffers[b].data();
    }
    Eigen::VectorXd residual(rows);
    Check(cost->Evaluate(parameters.data(),residual.data(),mode==4?nullptr:jacobians.data()),"factor Evaluate");
    Near(native.residual,residual,1e-10,"factor residual");
    for (std::size_t b=0;b<sizes.size();++b) if (jacobians[b]) {
      const Eigen::Map<splbatch::DynamicRowMajorMatrix> J(jacobians[b],rows,sizes[b]);
      Check(J.allFinite(),"Partial Jacobian must initialize every requested entry");
      const int col=b<4?6*b:b<8?6*(b-4)+3:24+3*(b-8);
      Near(native.jacobian.middleCols(col,3),J.leftCols(3),1e-10,"factor Jacobian");
      if (sizes[b]==4) Check(J.col(3).norm()==0,"Rotation padding column must be zero");
    }
  }
  // Finite differences perturb SO3 with the actual right-multiplicative update.
  Eigen::MatrixXd fd(rows,native.jacobian.cols());
  for (int c=0;c<fd.cols();++c) {
    Eigen::VectorXd delta=Eigen::VectorXd::Zero(s.initial.Dimension());delta[c]=1e-7;
    auto plus=s.initial,minus=s.initial;sc::ApplyIncrement(plus,delta);sc::ApplyIncrement(minus,-delta);
    const auto rp=imu?sc::NativeImuFactor(s,plus,s.imu[0],false).residual:sc::NativeLidarFactor(s,plus,s.lidar[0],false).residual;
    const auto rm=imu?sc::NativeImuFactor(s,minus,s.imu[0],false).residual:sc::NativeLidarFactor(s,minus,s.lidar[0],false).residual;
    fd.col(c)=(rp-rm)/(2e-7);
  }
  Near(native.jacobian,fd,2e-7,"finite difference");
}
void WholeProblemTest(int threads,bool with_prior) {
  auto s=sc::MakeSyntheticSnapshot(8,100,20,99);if (with_prior) AddPrior(s);s.Validate();
  const auto a=sc::LinearizeNative(s,s.initial,threads);
  for (auto backend:{sc::Backend::CeresScalar,sc::Backend::CeresBatch}) {
    const auto b=sc::LinearizeCeres(s,s.initial,backend,threads);
    Near(a.residual,b.residual,1e-10,"whole residual");Near(a.jacobian,b.jacobian,1e-10,"whole Jacobian");
    Near(a.jacobian.transpose()*a.jacobian,b.jacobian.transpose()*b.jacobian,1e-9,"whole Hessian");
    Near(a.jacobian.transpose()*a.residual,b.jacobian.transpose()*b.residual,1e-9,"whole gradient");
  }
  for (int iterations:{1,8}) {
    sc::Options options;options.threads=threads;options.iterations=iterations;options.backend=sc::Backend::CeresScalar;
    const auto b=sc::Solve(s,options);options.backend=sc::Backend::CeresBatch;const auto c=sc::Solve(s,options);
    Check(b.metrics.usable && c.metrics.usable,"B/C solve usable");
    Check(sc::StateDistance(b.state,c.state)<2e-7,"B/C solved state");
    Check(std::abs(b.metrics.final_cost-c.metrics.final_cost)<1e-7*std::max(1.,b.metrics.final_cost),"B/C solved cost");
    Check(c.metrics.residual_blocks<b.metrics.residual_blocks,"C must actually batch observations");
    options.backend=sc::Backend::Native;const auto a_result=sc::Solve(s,options);
    Check(a_result.metrics.usable && a_result.metrics.final_cost<a_result.metrics.initial_cost,"A solve decreases cost");
    Check((a_result.state.positions[0]-s.initial.positions[0]).norm()<1e-12,"A fixed position");
    Check((a_result.state.rotations[0].inverse()*s.initial.rotations[0]).log().norm()<1e-12,"A fixed rotation");
  }
}
void PriorAndSnapshotTest() {
  auto s=sc::MakeSyntheticSnapshot(9,120,30,77);AddPrior(s);
  const auto prior=sc::LinearizePrior(s.prior,s.initial);
  Eigen::MatrixXd fd(prior.jacobian.rows(),s.initial.Dimension());
  for (int c=0;c<fd.cols();++c) {
    Eigen::VectorXd d=Eigen::VectorXd::Zero(fd.cols());d[c]=1e-7;
    auto p=s.initial,m=s.initial;sc::ApplyIncrement(p,d);sc::ApplyIncrement(m,-d);
    fd.col(c)=(sc::LinearizePrior(s.prior,p).residual-sc::LinearizePrior(s.prior,m).residual)/(2e-7);
  }
  Near(prior.jacobian,fd,2e-8,"prior finite difference");
  auto flipped=s.initial;for (auto& rotation:flipped.rotations) {Eigen::Quaterniond q=rotation.unit_quaternion();q.coeffs()*=-1;rotation=sc::Rotation(q);}
  Near(prior.residual,sc::LinearizePrior(s.prior,flipped).residual,1e-12,"prior quaternion sign");
  auto p=sc::Marginalize(s,s.initial,2,2);
  Check(!p.Empty(),"generated prior");
  // Independently eliminate outgoing variables with a rank-revealing QR and
  // compare the quadratic cost change, including its linear term/sign.
  auto outgoing=s;
  outgoing.imu.erase(std::remove_if(outgoing.imu.begin(),outgoing.imu.end(),[](const auto& o){return o.span>=2;}),outgoing.imu.end());
  outgoing.lidar.erase(std::remove_if(outgoing.lidar.begin(),outgoing.lidar.end(),[](const auto& o){return o.span>=2;}),outgoing.lidar.end());
  const auto full=sc::LinearizeNative(outgoing,s.initial);
  const Eigen::MatrixXd Jm=full.jacobian.leftCols(12);
  Eigen::MatrixXd Jk(full.residual.size(),p.sqrt_information.cols());
  for (std::size_t i=0;i<p.knots.size();++i) Jk.middleCols(6*i,6)=full.jacobian.middleCols(6*(p.knots[i]+2),6);
  Jk.rightCols(6)=full.jacobian.rightCols(6);
  const auto qr=Jm.completeOrthogonalDecomposition();
  const Eigen::VectorXd baseline=full.residual-Jm*qr.solve(full.residual);
  for (int trial=0;trial<3;++trial) {
    Eigen::VectorXd delta(p.sqrt_information.cols());
    for (int c=0;c<delta.size();++c) delta[c]=1e-4*std::sin(3*c+trial+.5);
    const Eigen::VectorXd raw=full.residual+Jk*delta;
    const Eigen::VectorXd reduced=raw-Jm*qr.solve(raw);
    const double reference=reduced.squaredNorm()-baseline.squaredNorm();
    const double actual=(p.offset+p.sqrt_information*delta).squaredNorm()-p.offset.squaredNorm();
    Check(std::abs(actual-reference)<1e-7*std::max(1.,std::abs(reference)),"Schur prior versus independent QR elimination");
  }
  sc::Snapshot next=s;
  next.initial.rotations.erase(next.initial.rotations.begin(),next.initial.rotations.begin()+2);
  next.initial.positions.erase(next.initial.positions.begin(),next.initial.positions.begin()+2);
  auto advance=[](auto& observations) {observations.erase(std::remove_if(observations.begin(),observations.end(),[](const auto& o){return o.span<2;}),observations.end());for (auto& o:observations) o.span-=2;};
  advance(next.imu);advance(next.lidar);next.fixed_knots.clear();next.prior=p;next.Validate();
  const auto a=sc::LinearizeNative(next,next.initial);
  for (auto backend:{sc::Backend::CeresScalar,sc::Backend::CeresBatch}) {
    const auto b=sc::LinearizeCeres(next,next.initial,backend);
    Near(a.jacobian.transpose()*a.jacobian,b.jacobian.transpose()*b.jacobian,1e-9,"marginal prior Hessian");
    Near(a.jacobian.transpose()*a.residual,b.jacobian.transpose()*b.residual,1e-9,"marginal prior gradient");
  }
  const auto directory=std::filesystem::temp_directory_path()/("slict_abc_test_"+std::to_string(getpid()));
  std::filesystem::create_directories(directory);const auto path=(directory/"window.slict").string();
  sc::SaveSnapshot(next,path);const auto read=sc::LoadSnapshot(path);
  Check(sc::SnapshotDigest(next)==sc::SnapshotDigest(read),"snapshot round trip");
  std::fstream file(path,std::ios::in|std::ios::out|std::ios::binary);file.seekp(36);file.put('\x7f');file.close();
  bool rejected=false;try {sc::LoadSnapshot(path);} catch (const std::exception&) {rejected=true;}Check(rejected,"corrupt snapshot rejected");
  std::filesystem::remove_all(directory);
  s.lidar[0].span=10000;rejected=false;try {s.Validate();}catch (const std::exception&){rejected=true;}Check(rejected,"invalid support rejected");
}
void ContinuedCeresStepsTest() {
  auto b_input=sc::MakeSyntheticSnapshot(8,100,20,42),c_input=b_input;
  sc::Options b_options,c_options;
  b_options.backend=sc::Backend::CeresScalar;c_options.backend=sc::Backend::CeresBatch;
  b_options.ceres_initial_trust_region_radius=c_options.ceres_initial_trust_region_radius=200;
  for (int i=0;i<6;++i) {
    const auto b=sc::Solve(b_input,b_options),c=sc::Solve(c_input,c_options);
    Check(b.metrics.usable && c.metrics.usable,"continued B/C step usable");
    Check(b.metrics.iterations<=1 && c.metrics.iterations<=1,"continued calls remain single step");
    Check(sc::StateDistance(b.state,c.state)<2e-7,"continued B/C state agreement");
    Check(b.metrics.ceres_final_trust_region_radius>0 && c.metrics.ceres_final_trust_region_radius>0,
          "Ceres returns continuation radius");
    b_input.initial=b.state;c_input.initial=c.state;
    b_options.ceres_initial_trust_region_radius=b.metrics.ceres_final_trust_region_radius;
    c_options.ceres_initial_trust_region_radius=c.metrics.ceres_final_trust_region_radius;
  }
}

void NativeBatchTest() {
  for (int threads:{1,4}) for (int mode=0;mode<5;++mode) {
    auto s=sc::MakeSyntheticSnapshot(9,mode==1?0:120,mode==2?0:30,77);
    s.fixed_knots={0,3,8};
    AddPrior(s);
    if (mode==3) {s.imu.clear();s.lidar.clear();}
    if (mode==4) {s.imu.clear();s.lidar.clear();s.prior={};}
    for (int i=0;i<9;i+=2) {
      Eigen::Quaterniond q=s.initial.rotations[i].unit_quaternion();q.coeffs()*=-1;
      s.initial.rotations[i]=sc::Rotation(q);
    }
    s.Validate();
    auto state=s.initial;
    auto batch=sc::BuildNativeBatchProblem(s,state);
    // Reuse the exact registered pointers after state changes; runtime data
    // must be recomputed. Interleave residual-only and Jacobian evaluations.
    for (int trial=0;trial<2;++trial) {
      const auto ref=sc::LinearizeNative(s,state,threads);
      Eigen::VectorXd residual;Eigen::MatrixXd jacobian;
      batch->Evaluate(threads,residual,nullptr);
      Near(ref.residual,residual,1e-10,"native batch residual only");
      batch->Evaluate(threads,residual,&jacobian);
      Near(ref.residual,residual,1e-10,"native batch residual");
      Near(ref.jacobian,jacobian,1e-10,"native batch tangent scatter and constants");
      if (trial==0) {
        Eigen::VectorXd delta=Eigen::VectorXd::LinSpaced(state.Dimension(),-.002,.003);
        sc::ApplyIncrement(state,delta);
      }
    }
    for (int iterations:{1,8}) for (double clip:{.0001,0.5,0.0}) {
      sc::Options o;o.threads=threads;o.iterations=iterations;o.step_limit=clip;
      const auto a=sc::Solve(s,o);o.backend=sc::Backend::NativeBatch;const auto d=sc::Solve(s,o);
      Check(a.metrics.usable && d.metrics.usable,"native/batch usable");
      Check(sc::StateDistance(a.state,d.state)<1e-7,"native/batch clipped or unclipped solve state");
      Check(std::abs(a.metrics.final_cost-d.metrics.final_cost)<1e-8*std::max(1.,a.metrics.final_cost),"native/batch solve cost");
      for (int knot:s.fixed_knots) {
        Check((d.state.positions[knot]-s.initial.positions[knot]).norm()<1e-12,"native batch fixed position");
        Check((d.state.rotations[knot].inverse()*s.initial.rotations[knot]).log().norm()<1e-12,"native batch fixed rotation");
      }
      Check(d.metrics.ceres_final_trust_region_radius==0,"native batch must not use Ceres optimizer");
      if (mode==0) Check(d.metrics.residual_blocks<a.metrics.residual_blocks,"native batch groups observations");
    }
  }
  // A missing canonical sort may permute rows; the represented normal system
  // must still agree. This also exercises support-dependent column mappings.
  auto s=sc::MakeSyntheticSnapshot(9,120,30,81);
  std::reverse(s.imu.begin(),s.imu.end());std::reverse(s.lidar.begin(),s.lidar.end());
  const auto a=sc::LinearizeNative(s,s.initial,4),d=sc::LinearizeNativeBatch(s,s.initial,4);
  Near(a.jacobian.transpose()*a.jacobian,d.jacobian.transpose()*d.jacobian,1e-9,"unsorted native batch H");
  Near(a.jacobian.transpose()*a.residual,d.jacobian.transpose()*d.residual,1e-9,"unsorted native batch g");
}

void RegistryCompatibilityTest() {
  struct Registry final:splbatch::ResidualRegistry {
    int count=0;
    std::vector<std::unique_ptr<ceres::CostFunction>> costs;
    void AddResidualBlock(std::unique_ptr<ceres::CostFunction> cost,std::vector<double*> parameters) override {
      Check(cost->parameter_block_sizes().size()==parameters.size(),"registry parameter layout");
      ++count;costs.push_back(std::move(cost));
    }
  } registry;
  auto s=sc::MakeSyntheticSnapshot(4,2,0,42);
  splbatch::ProblemBatcher batcher(registry);
  auto& lidar=batcher.RegisterEvaluator(sc::LidarEvaluator());
  for (const auto& o:s.lidar) lidar.AddObservation(sc::Bind(s.initial,o.span,s.dt,false),o);
  batcher.Commit();batcher.Commit();Check(registry.count==1,"registry commit once");
  bool rejected=false;
  try {lidar.AddObservation(sc::Bind(s.initial,0,s.dt,false),s.lidar[0]);}
  catch (const std::logic_error&) {rejected=true;}
  Check(rejected,"registry seals observations");
  rejected=false;ceres::Solver::Summary summary;
  try {batcher.Solve(ceres::Solver::Options(),&summary);}
  catch (const std::logic_error&) {rejected=true;}
  Check(rejected,"native registry cannot invoke Ceres Solve");
  // Preserve direct channel CommitTo(ceres::Problem&) compatibility as well.
  ceres::Problem problem;
  splbatch::EvaluatorChannel<sc::LidarEvaluator> channel(sc::LidarEvaluator(),splbatch::NoLossPolicy{});
  channel.AddObservation(sc::Bind(s.initial,0,s.dt,false),s.lidar[0]);
  channel.CommitTo(problem);channel.CommitTo(problem);
  Check(problem.NumResidualBlocks()==1,"legacy channel Ceres registration");
}

void InlineBindingContextTest() {
  auto s=sc::MakeSyntheticSnapshot(4,0,3,41);s.fixed_knots.clear();
  for (bool imu:{false,true}) {
    const auto owned=sc::Bind(s.initial,0,s.dt,imu);
    const auto input=sc::BindInput(s.initial,0,s.dt,imu);
    Check(owned.Key()==input.ToBinding().Key(),"SLICT inline input changed parameter layout");
    Check(input.parameters.blocks.IsInline() && input.parameters.sizes.IsInline(),
          "SLICT's 8/10 parameter blocks spilled");
  }
  ceres::Problem problem;
  splbatch::ProblemBatcher batcher(problem);
  auto& channel=batcher.RegisterEvaluator(sc::ImuEvaluator(s));
  Eigen::VectorXd expected(36);
  for (int i=0;i<3;++i) {
    auto current=s;
    current.dt=s.dt*(i==1?1.5:1.);
    const auto& o=s.imu[i];
    channel.AddObservation(sc::BindInput(s.initial,o.span,current.dt,true),o);
    expected.segment<12>(12*i)=sc::NativeImuFactor(current,s.initial,o,false).residual;
  }
  Check(channel.PendingBatchCount()==1,"Changing dt unexpectedly changed parameter support");
  batcher.Commit();
  std::vector<ceres::ResidualBlockId> ids;problem.GetResidualBlocks(&ids);
  Check(ids.size()==1,"Context test batch count");
  std::vector<double*> parameters;problem.GetParameterBlocksForResidualBlock(ids[0],&parameters);
  Eigen::VectorXd residual(36);
  Check(problem.GetCostFunctionForResidualBlock(ids[0])->Evaluate(parameters.data(),residual.data(),nullptr),
        "Context test Evaluate");
  Near(residual,expected,1e-10,"Inline IMU dt change and restore");
}
}
int main() {
  try {
    Eigen::setNbThreads(1);
    for (auto backend:{sc::Backend::CeresScalar,sc::Backend::CeresBatch})
      for (int situation=0;situation<3;++situation) {FactorTest(false,situation,backend);FactorTest(true,situation,backend);}
    for (int threads:{1,4}) {WholeProblemTest(threads,false);WholeProblemTest(threads,true);}
    PriorAndSnapshotTest();ContinuedCeresStepsTest();NativeBatchTest();RegistryCompatibilityTest();InlineBindingContextTest();
    std::cout<<"SLICT A/B/C/D factor, prior, registry, solve, threading and snapshot tests passed\n";return 0;
  } catch (const std::exception& e) {std::cerr<<"FAILED: "<<e.what()<<'\n';return 1;}
}
