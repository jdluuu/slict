#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "splbatch/jacobian_writer.hpp"

namespace splbatch
{
  /// Construct A and b such that A^T A = H and A^T b = g.
  ///
  /// Structurally zero ambient columns are removed before attempting LLT.
  /// This keeps quaternion-style 4D parameter blocks efficient when their
  /// evaluator Jacobian uses only three tangent columns. A rank-revealing
  /// eigendecomposition handles remaining singular normal equations.
  inline bool MakeSquareRootNormalEquation(
      const Eigen::MatrixXd &hessian,
      const Eigen::VectorXd &gradient,
      DynamicRowMajorMatrix &square_root,
      Eigen::VectorXd &compact_residual)
  {
    const Eigen::Index dimension = hessian.rows();
    if (dimension <= 0 || hessian.cols() != dimension ||
        gradient.size() != dimension || !hessian.allFinite() ||
        !gradient.allFinite())
    {
      return false;
    }

    const Eigen::MatrixXd symmetric_hessian =
        0.5 * (hessian + hessian.transpose());
    square_root.setZero(dimension, dimension);
    compact_residual.setZero(dimension);

    const double spectral_scale = std::max(
        1.0,
        symmetric_hessian.cwiseAbs().rowwise().sum().maxCoeff());
    const double rank_tolerance =
        64.0 * static_cast<double>(dimension) *
        std::numeric_limits<double>::epsilon() * spectral_scale;
    const double negative_tolerance = 1e-11 * spectral_scale;
    // Ill-conditioned, rank-deficient measurement factors can leak a small
    // gradient into a numerically null eigendirection. Keep the consistency
    // check relative to the accumulated gradient scale while allowing for
    // the H=J^T J eigensolve and projection roundoff.
    const double gradient_tolerance =
        1e-7 * std::max(1.0, gradient.cwiseAbs().maxCoeff());

    std::vector<Eigen::Index> active_columns;
    active_columns.reserve(static_cast<std::size_t>(dimension));
    for (Eigen::Index column = 0; column < dimension; ++column)
    {
      const double diagonal = symmetric_hessian(column, column);
      if (diagonal < -negative_tolerance)
        return false;
      if (diagonal > rank_tolerance)
      {
        active_columns.push_back(column);
      }
      else if (std::abs(gradient[column]) > gradient_tolerance)
      {
        return false;
      }
    }

    const Eigen::Index active_dimension =
        static_cast<Eigen::Index>(active_columns.size());
    if (active_dimension == 0)
      return true;

    Eigen::MatrixXd active_hessian(active_dimension, active_dimension);
    Eigen::VectorXd active_gradient(active_dimension);
    for (Eigen::Index row = 0; row < active_dimension; ++row)
    {
      active_gradient[row] = gradient[active_columns[row]];
      for (Eigen::Index column = 0; column < active_dimension; ++column)
      {
        active_hessian(row, column) = symmetric_hessian(
            active_columns[row], active_columns[column]);
      }
    }

    const auto scatter_square_root =
        [&](const Eigen::MatrixXd &active_square_root,
            const Eigen::VectorXd &active_residual)
    {
      const Eigen::Index row_count = active_square_root.rows();
      compact_residual.head(row_count) = active_residual;
      for (Eigen::Index local_column = 0;
           local_column < active_dimension; ++local_column)
      {
        square_root.block(
            0, active_columns[local_column], row_count, 1) =
            active_square_root.col(local_column);
      }
    };

    Eigen::LLT<Eigen::MatrixXd> llt(active_hessian);
    if (llt.info() == Eigen::Success)
    {
      const Eigen::MatrixXd lower = llt.matrixL();
      if (lower.diagonal().cwiseAbs().minCoeff() >
          std::sqrt(rank_tolerance))
      {
        const Eigen::MatrixXd active_square_root = lower.transpose();
        const Eigen::VectorXd active_residual =
            lower.template triangularView<Eigen::Lower>().solve(
                active_gradient);
        scatter_square_root(active_square_root, active_residual);
        return active_residual.allFinite();
      }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(active_hessian);
    if (solver.info() != Eigen::Success ||
        solver.eigenvalues().minCoeff() < -negative_tolerance)
    {
      return false;
    }

    Eigen::MatrixXd active_square_root =
        Eigen::MatrixXd::Zero(active_dimension, active_dimension);
    Eigen::VectorXd active_residual =
        Eigen::VectorXd::Zero(active_dimension);
    Eigen::Index rank = 0;
    for (Eigen::Index eigen_index = 0;
         eigen_index < active_dimension; ++eigen_index)
    {
      const double eigenvalue = solver.eigenvalues()[eigen_index];
      const double projected_gradient =
          solver.eigenvectors().col(eigen_index).dot(active_gradient);
      if (eigenvalue <= rank_tolerance &&
          std::abs(projected_gradient) <= gradient_tolerance)
      {
        continue;
      }
      // A small positive mode may still carry a meaningful gradient in an
      // ill-conditioned range/bearing normal equation.  Retain that mode;
      // only numerically non-positive modes are impossible to square-root.
      if (eigenvalue <= 0.0)
        return false;

      const double square_root_eigenvalue = std::sqrt(eigenvalue);
      active_square_root.row(rank) =
          square_root_eigenvalue *
          solver.eigenvectors().col(eigen_index).transpose();
      active_residual[rank] =
          projected_gradient / square_root_eigenvalue;
      ++rank;
    }

    scatter_square_root(
        active_square_root.topRows(rank), active_residual.head(rank));
    return square_root.allFinite() && compact_residual.allFinite();
  }

  /// Fixed-size counterpart used when an evaluator declares a compile-time
  /// normal-group shape. It avoids active-column discovery, dynamic matrix
  /// copies, and dynamic LLT dispatch. Structural zero columns have already
  /// been removed by CompiledJacobianStructure; a fixed-size eigensystem is
  /// retained as the rank-deficient fallback.
  template <int Dimension>
  inline bool MakeFixedSquareRootNormalEquation(
      const Eigen::Matrix<double, Dimension, Dimension> &hessian,
      const Eigen::Matrix<double, Dimension, 1> &gradient,
      Eigen::Matrix<double, Dimension, Dimension, Eigen::RowMajor>
          &square_root,
      Eigen::Matrix<double, Dimension, 1> &compact_residual)
  {
    static_assert(Dimension > 0, "Normal dimension must be positive");
    using Matrix = Eigen::Matrix<double, Dimension, Dimension>;

    if (!hessian.allFinite() || !gradient.allFinite())
      return false;
    const Matrix symmetric_hessian =
        0.5 * (hessian + hessian.transpose());
    square_root.setZero();
    compact_residual.setZero();

    const double spectral_scale = std::max(
        1.0,
        symmetric_hessian.cwiseAbs().rowwise().sum().maxCoeff());
    const double rank_tolerance =
        64.0 * static_cast<double>(Dimension) *
        std::numeric_limits<double>::epsilon() * spectral_scale;
    const double negative_tolerance = 1e-11 * spectral_scale;
    const double gradient_tolerance =
        1e-7 * std::max(1.0, gradient.cwiseAbs().maxCoeff());

    Eigen::LLT<Matrix> llt(symmetric_hessian);
    if (llt.info() == Eigen::Success)
    {
      const Matrix lower = llt.matrixL();
      if (lower.diagonal().cwiseAbs().minCoeff() >
          std::sqrt(rank_tolerance))
      {
        square_root = lower.transpose();
        compact_residual =
            lower.template triangularView<Eigen::Lower>().solve(gradient);
        return square_root.allFinite() && compact_residual.allFinite();
      }
    }

    Eigen::SelfAdjointEigenSolver<Matrix> solver(symmetric_hessian);
    if (solver.info() != Eigen::Success ||
        solver.eigenvalues().minCoeff() < -negative_tolerance)
    {
      return false;
    }

    Eigen::Index rank = 0;
    for (Eigen::Index eigen_index = 0;
         eigen_index < Dimension; ++eigen_index)
    {
      const double eigenvalue = solver.eigenvalues()[eigen_index];
      const double projected_gradient =
          solver.eigenvectors().col(eigen_index).dot(gradient);
      if (eigenvalue <= rank_tolerance &&
          std::abs(projected_gradient) <= gradient_tolerance)
      {
        continue;
      }
      if (eigenvalue <= 0.0)
        return false;
      const double square_root_eigenvalue = std::sqrt(eigenvalue);
      square_root.row(rank) =
          square_root_eigenvalue *
          solver.eigenvectors().col(eigen_index).transpose();
      compact_residual[rank] =
          projected_gradient / square_root_eigenvalue;
      ++rank;
    }
    return square_root.allFinite() && compact_residual.allFinite();
  }
} // namespace splbatch
