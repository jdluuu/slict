#include "internal.h"
#include "evaluators.h"
#include "basalt/spline/ceres_local_param.hpp"
#include <splbatch/problem_batcher.hpp>
#include <numeric>
#include <set>

namespace slict::comparison {
namespace {
template <typename Evaluator> class ScalarCost final : public ceres::CostFunction {
 public:
  ScalarCost(Evaluator evaluator, Binding binding, const typename Evaluator::Observation& observation)
      : evaluator_(std::move(evaluator)), binding_(std::move(binding)),
        prepared_(evaluator_.PrepareObservation(binding_,observation)) {
    set_num_residuals(Evaluator::kResidualDim);
    *mutable_parameter_block_sizes()=binding_.parameters.sizes;
  }
  bool Evaluate(double const* const* p,double* residuals,double** jacobians) const override {
    if (jacobians) for (std::size_t i=0;i<parameter_block_sizes().size();++i)
      if (jacobians[i]) std::fill_n(jacobians[i],num_residuals()*parameter_block_sizes()[i],0.0);
    splbatch::EvaluationRequest request(jacobians,parameter_block_sizes().size());
    const auto runtime=evaluator_.PrepareBatch(binding_,p,request);
    const splbatch::ObservationOutput out(residuals,jacobians,parameter_block_sizes(),0,num_residuals());
    return evaluator_.EvaluateObservation(binding_,prepared_,runtime,p,out);
  }
 private:
  Evaluator evaluator_;
  Binding binding_;
  typename Evaluator::PreparedObservation prepared_;
};
class PriorCost final : public ceres::CostFunction {
 public:
  explicit PriorCost(const Prior& prior):prior_(prior) {
    std::iota(prior_.knots.begin(),prior_.knots.end(),0);
    set_num_residuals(prior.offset.size());
    for (std::size_t i=0;i<prior.knots.size();++i) {
      mutable_parameter_block_sizes()->push_back(4);
      mutable_parameter_block_sizes()->push_back(3);
    }
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);
  }
  bool Evaluate(double const* const* p,double* residuals,double** jacobians) const override {
    State state; const int knots=prior_.knots.size();
    state.rotations.resize(knots);state.positions.resize(knots);
    for (int i=0;i<knots;++i) {
      state.rotations[i]=Eigen::Map<const Rotation>(p[2*i]);
      state.positions[i]=Eigen::Map<const Vector3>(p[2*i+1]);
    }
    state.gyro_bias=Eigen::Map<const Vector3>(p[2*knots]);
    state.accel_bias=Eigen::Map<const Vector3>(p[2*knots+1]);
    auto value=LinearizePrior(prior_,state);
    Eigen::Map<Eigen::VectorXd>(residuals,num_residuals())=value.residual;
    if (jacobians) for (int b=0;b<2*knots+2;++b) if (jacobians[b]) {
      Eigen::Map<splbatch::DynamicRowMajorMatrix> J(jacobians[b],num_residuals(),parameter_block_sizes()[b]);
      J.setZero();J.leftCols(3)=value.jacobian.middleCols(3*b,3);
    }
    return value.residual.allFinite();
  }
 private: Prior prior_;
};
}

CeresProblem BuildCeresProblem(const Snapshot& s,State& state,Backend backend) {
  CeresProblem bundle;bundle.problem=std::make_unique<ceres::Problem>();
  auto& problem=*bundle.problem;
  auto* rotation_parameterization=new basalt::LieAnalyticLocalParameterization<Rotation>();
  const std::set<int> fixed(s.fixed_knots.begin(),s.fixed_knots.end());
  for (std::size_t i=0;i<state.rotations.size();++i) {
    auto* r=state.rotations[i].data();auto* p=state.positions[i].data();
    problem.AddParameterBlock(r,4,rotation_parameterization);problem.AddParameterBlock(p,3);
    if (fixed.count(i)) {problem.SetParameterBlockConstant(r);problem.SetParameterBlockConstant(p);}
    else {
      bundle.parameters.push_back(r);bundle.parameters.push_back(p);
      for (int c=0;c<6;++c) bundle.full_column_indices.push_back(6*i+c);
    }
  }
  problem.AddParameterBlock(state.gyro_bias.data(),3);problem.AddParameterBlock(state.accel_bias.data(),3);
  bundle.parameters.push_back(state.gyro_bias.data());bundle.parameters.push_back(state.accel_bias.data());
  for (int c=0;c<6;++c) bundle.full_column_indices.push_back(state.Dimension()-6+c);
  if (backend==Backend::CeresScalar) {
    for (const auto& o:s.imu) {
      auto binding=Bind(state,o.span,s.dt,true);
      problem.AddResidualBlock(new ScalarCost<ImuEvaluator>(ImuEvaluator(s),binding,o),nullptr,binding.parameters.blocks);
    }
    for (const auto& o:s.lidar) {
      auto binding=Bind(state,o.span,s.dt,false);
      problem.AddResidualBlock(new ScalarCost<LidarEvaluator>(LidarEvaluator(),binding,o),nullptr,binding.parameters.blocks);
    }
  } else {
    splbatch::ProblemBatcher batcher(problem);
    auto& imu=batcher.RegisterEvaluator(ImuEvaluator(s));
    auto& lidar=batcher.RegisterEvaluator(LidarEvaluator());
    for (const auto& o:s.imu) imu.AddObservation(BindInput(state,o.span,s.dt,true),o);
    for (const auto& o:s.lidar) lidar.AddObservation(BindInput(state,o.span,s.dt,false),o);
    batcher.Commit();  // Includes grouping, preparation and registration in build_ms.
  }
  if (!s.prior.Empty()) {
    std::vector<double*> parameters;
    for (int i:s.prior.knots) {parameters.push_back(state.rotations[i].data());parameters.push_back(state.positions[i].data());}
    parameters.push_back(state.gyro_bias.data());parameters.push_back(state.accel_bias.data());
    problem.AddResidualBlock(new PriorCost(s.prior),nullptr,parameters);
  }
  return bundle;
}

Linearization LinearizeCeres(const Snapshot& s,const State& input,Backend backend,int threads) {
  State state=input;auto bundle=BuildCeresProblem(s,state,backend);
  ceres::Problem::EvaluateOptions options;options.num_threads=threads;options.apply_loss_function=true;
  options.parameter_blocks=bundle.parameters;
  ceres::CRSMatrix crs;std::vector<double> residuals;
  if (!bundle.problem->Evaluate(options,nullptr,&residuals,nullptr,&crs)) throw std::runtime_error("Ceres evaluation failed");
  Linearization result;
  result.residual=Eigen::Map<Eigen::VectorXd>(residuals.data(),residuals.size());
  result.jacobian=Eigen::MatrixXd::Zero(crs.num_rows,state.Dimension());
  for (int row=0;row<crs.num_rows;++row) for (int j=crs.rows[row];j<crs.rows[row+1];++j)
    result.jacobian(row,bundle.full_column_indices.at(crs.cols[j]))=crs.values[j];
  return result;
}

Result SolveCeres(const Snapshot& s,const Options& o) {
  const auto total=Clock::now();Result result;auto& m=result.metrics;
  auto phase=Clock::now();result.state=s.initial;m.reset_ms=Milliseconds(phase);
  phase=Clock::now();auto bundle=BuildCeresProblem(s,result.state,o.backend);m.build_ms=Milliseconds(phase);
  m.residual_blocks=bundle.problem->NumResidualBlocks();m.scalar_residuals=bundle.problem->NumResiduals();
  ceres::Solver::Options options;
  options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library_type=ceres::SUITE_SPARSE;
  options.trust_region_strategy_type=ceres::LEVENBERG_MARQUARDT;
  options.num_threads=o.threads;options.max_num_iterations=o.iterations;
  options.function_tolerance=o.function_tolerance;options.gradient_tolerance=o.gradient_tolerance;
  options.parameter_tolerance=o.parameter_tolerance;
  options.max_solver_time_in_seconds=1e9;options.minimizer_progress_to_stdout=false;
  options.use_nonmonotonic_steps=false;
  options.initial_trust_region_radius=o.ceres_initial_trust_region_radius;
  ceres::Solver::Summary summary;
  phase=Clock::now();ceres::Solve(options,bundle.problem.get(),&summary);m.solve_ms=Milliseconds(phase);
  m.usable=summary.IsSolutionUsable();m.termination=ceres::TerminationTypeToString(summary.termination_type);
  m.initial_cost=summary.initial_cost;m.final_cost=summary.final_cost;
  m.iterations=std::max(0,static_cast<int>(summary.iterations.size())-1);
  m.successful_steps=summary.num_successful_steps;m.unsuccessful_steps=summary.num_unsuccessful_steps;
  if (!summary.iterations.empty())
    m.ceres_final_trust_region_radius=summary.iterations.back().trust_region_radius;
  m.residual_evaluation_ms=1000*summary.residual_evaluation_time_in_seconds;
  m.jacobian_evaluation_ms=1000*summary.jacobian_evaluation_time_in_seconds;
  m.evaluate_ms=m.residual_evaluation_ms+m.jacobian_evaluation_ms;
  m.linear_ms=1000*summary.linear_solver_time_in_seconds;
  phase=Clock::now();bundle.problem.reset();m.destroy_ms=Milliseconds(phase);
  m.total_ms=Milliseconds(total);
  return result;
}
}  // namespace slict::comparison
