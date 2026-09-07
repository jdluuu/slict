#pragma once

#include <memory>
#include <vector>
#include <ceres/cost_function.h>
#include <ceres/problem.h>

namespace splbatch {
// Registration boundary only: evaluators, grouping and batch evaluation remain
// identical. A registry takes ownership and keeps parameter storage non-owning.
class ResidualRegistry {
 public:
  virtual ~ResidualRegistry() = default;
  virtual void AddResidualBlock(std::unique_ptr<ceres::CostFunction> cost,
                                std::vector<double*> parameters) = 0;
};

class CeresResidualRegistry final : public ResidualRegistry {
 public:
  explicit CeresResidualRegistry(ceres::Problem& problem) : problem_(problem) {}
  void AddResidualBlock(std::unique_ptr<ceres::CostFunction> cost,
                        std::vector<double*> parameters) override {
    problem_.AddResidualBlock(cost.get(), nullptr, parameters);
    cost.release();  // Same ownership convention as the existing Ceres path.
  }
 private:
  ceres::Problem& problem_;
};
}  // namespace splbatch
