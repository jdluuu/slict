#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "splbatch/jacobian_writer.hpp"

namespace splbatch
{
  /// Validate the outer scale used by ceres::ScaledLoss. CTRIO treats
  /// distance/bearing confidence weights as strictly positive, so the batch
  /// interface follows the same contract instead of silently dropping an
  /// observation with a zero or invalid scale.
  inline void ValidateObservationLossScale(const double loss_scale)
  {
    if (!std::isfinite(loss_scale) || loss_scale <= 0.0)
    {
      throw std::invalid_argument(
          "Observation loss scale must be finite and positive");
    }
  }

  class NoLossPolicy
  {
  public:
    int ExtraResidualRows() const { return 0; }

    template <typename Output>
    double Apply(const Output &output,
                 const double observation_loss_scale = 1.0) const
    {
      ValidateObservationLossScale(observation_loss_scale);
      double projection_gap = 0.0;
      if (output.HasUnprojectedSquaredNorm())
      {
        const double stored_norm = output.Residual().squaredNorm();
        projection_gap = std::max(
            0.0, output.UnprojectedSquaredNorm(stored_norm) - stored_norm);
      }
      if (observation_loss_scale != 1.0)
        output.Scale(std::sqrt(observation_loss_scale));
      return observation_loss_scale * projection_gap;
    }

    void Finalize(double *, const int, const double) const {}
  };

  /// Applies one Huber loss independently to every observation in a batch.
  /// Apply(output, w) is equivalent to ceres::ScaledLoss(Huber(delta), w):
  /// w multiplies rho, rho', and the robust cost gap without changing the
  /// Huber kernel's input or threshold.
  ///
  /// The outer Ceres residual block must use a null LossFunction.  One final
  /// zero-Jacobian residual stores the difference between rho(s) and rho'(s)s,
  /// preserving the same cost, gradient, and Gauss-Newton Hessian as separate
  /// Ceres residual blocks with individual HuberLoss instances.
  class IndependentHuberLossPolicy
  {
  public:
    explicit IndependentHuberLossPolicy(const double delta) : delta_(delta)
    {
      if (!std::isfinite(delta_) || delta_ < 0.0)
      {
        throw std::invalid_argument(
            "Huber delta must be finite and nonnegative");
      }
    }

    int ExtraResidualRows() const { return 1; }

    template <typename Output>
    double Apply(const Output &output,
                 const double observation_loss_scale = 1.0) const
    {
      ValidateObservationLossScale(observation_loss_scale);
      const double stored_squared_norm = output.Residual().squaredNorm();
      const double squared_norm = output.UnprojectedSquaredNorm(stored_squared_norm);
      double rho0 = squared_norm;
      double rho1 = 1.0;
      if (delta_ > 0.0)
      {
        const double delta_squared = delta_ * delta_;
        if (squared_norm > delta_squared)
        {
          const double norm = std::sqrt(squared_norm);
          rho0 = 2.0 * delta_ * norm - delta_squared;
          rho1 = delta_ / norm;
        }
      }

      output.Scale(std::sqrt(observation_loss_scale * rho1));
      return std::max(
          0.0, observation_loss_scale *
                   (rho0 - rho1 * stored_squared_norm));
    }

    void Finalize(double *residuals, const int data_row_count,
                  const double accumulated_cost_gap) const
    {
      residuals[data_row_count] =
          std::sqrt(std::max(0.0, accumulated_cost_gap));
    }

  private:
    double delta_;
  };
} // namespace splbatch
