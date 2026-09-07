#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "splbatch/jacobian_structure.hpp"

namespace splbatch
{
  using DynamicRowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  /// One structurally nonzero rectangle inside an observation Jacobian block.
  /// Compressed factors use these slices to avoid touching the zero rows and
  /// columns when a loss policy scales an observation linearization.
  struct ActiveJacobianSlice
  {
    std::size_t block = 0;
    int row_offset = 0;
    int row_count = 0;
    int column_offset = 0;
    int column_count = 0;
  };

  /// Storage-independent scatter/zero plan, compiled once per support. Unlike
  /// the compression plan it does not construct normal-equation components.
  class JacobianWritePlan
  {
  public:
    JacobianWritePlan(const JacobianStructure &structure, const int rows,
                      const std::vector<int32_t> &sizes)
    {
      std::vector<bool> owned(rows, false);
      std::vector<std::vector<bool>> active;
      for (const int size : sizes)
        active.emplace_back(rows * size, false);
      for (const auto &group : structure.groups)
      {
        const auto range = group.residual_rows;
        if (range.offset < 0 || range.count <= 0 ||
            static_cast<int64_t>(range.offset) + range.count > rows)
          throw std::invalid_argument("Invalid Jacobian row range");
        for (int r = range.offset; r < range.offset + range.count; ++r)
        {
          if (owned[r])
            throw std::invalid_argument("Overlapping Jacobian row ranges");
          owned[r] = true;
        }
        for (const auto &slice : group.parameter_slices)
        {
          if (slice.block >= sizes.size() || slice.column_offset < 0 ||
              slice.column_count <= 0 ||
              static_cast<int64_t>(slice.column_offset) + slice.column_count > sizes[slice.block])
            throw std::invalid_argument("Invalid Jacobian column slice");
          active_slices.push_back({slice.block, range.offset, range.count,
                                   slice.column_offset, slice.column_count});
          for (int r = range.offset; r < range.offset + range.count; ++r)
            for (int c = slice.column_offset; c < slice.column_offset + slice.column_count; ++c)
            {
              const int index = r * sizes[slice.block] + c;
              if (active[slice.block][index])
                throw std::invalid_argument("Overlapping Jacobian slices");
              active[slice.block][index] = true;
            }
        }
      }
      for (const bool row_owned : owned)
        if (!row_owned)
          throw std::invalid_argument("Jacobian rows must form a partition");
      for (std::size_t block = 0; block < sizes.size(); ++block)
        for (int r = 0; r < rows; ++r)
          for (int c = 0; c < sizes[block];)
          {
            if (active[block][r * sizes[block] + c]) { ++c; continue; }
            const int start = c;
            while (c < sizes[block] && !active[block][r * sizes[block] + c]) ++c;
            if (!zero_slices.empty() && zero_slices.back().block == block &&
                zero_slices.back().column_offset == start &&
                zero_slices.back().column_count == c - start &&
                zero_slices.back().row_offset + zero_slices.back().row_count == r)
              ++zero_slices.back().row_count;
            else
              zero_slices.push_back({block, r, 1, start, c - start});
          }
    }

    std::vector<ActiveJacobianSlice> active_slices;
    std::vector<ActiveJacobianSlice> zero_slices;
  };

  /// Optional raw residual norm for a locally projected residual/Jacobian.
  /// Projection drops only components orthogonal to J's column space; the
  /// loss must still see the original norm. Set before applying a loss.
  class ResidualProjection
  {
  public:
    void SetUnprojectedSquaredNorm(const double value) const { raw_squared_norm_ = value; }
    bool HasUnprojectedSquaredNorm() const { return raw_squared_norm_ >= 0.0; }
    double UnprojectedSquaredNorm(const double stored_squared_norm) const
    {
      return raw_squared_norm_ < 0.0 ? stored_squared_norm : raw_squared_norm_;
    }
  private:
    mutable double raw_squared_norm_ = -1.0;
  };

  class ObservationOutput : public ResidualProjection
  {
  public:
    ObservationOutput(double *residuals, double *const *jacobians,
                      const std::vector<int32_t> &parameter_block_sizes,
                      const int row_offset, const int residual_dimension,
                      const std::vector<ActiveJacobianSlice>
                          *active_jacobian_slices = nullptr)
        : residuals_(residuals), jacobians_(jacobians),
          parameter_block_sizes_(&parameter_block_sizes),
          row_offset_(row_offset), residual_dimension_(residual_dimension),
          active_jacobian_slices_(active_jacobian_slices)
    {
      assert(residuals_ != nullptr);
      assert(row_offset_ >= 0);
      assert(residual_dimension_ > 0);
    }

    Eigen::Map<Eigen::VectorXd> Residual() const
    {
      return Eigen::Map<Eigen::VectorXd>(
          residuals_ + row_offset_, residual_dimension_);
    }

    bool JacobianRequested(const std::size_t block) const
    {
      return jacobians_ != nullptr &&
             block < parameter_block_sizes_->size() &&
             jacobians_[block] != nullptr;
    }

    Eigen::Map<DynamicRowMajorMatrix> Jacobian(
        const std::size_t block) const
    {
      assert(JacobianRequested(block));
      const int32_t block_size = (*parameter_block_sizes_)[block];
      return Eigen::Map<DynamicRowMajorMatrix>(
          jacobians_[block] + row_offset_ * block_size,
          residual_dimension_, block_size);
    }

    std::size_t ParameterBlockCount() const
    {
      return parameter_block_sizes_->size();
    }

    void Scale(const double scale) const
    {
      if (scale == 1.0)
        return;
      for (int row = 0; row < residual_dimension_; ++row)
        residuals_[row_offset_ + row] *= scale;
      if (jacobians_ == nullptr)
        return;
      if (active_jacobian_slices_ != nullptr)
      {
        for (const ActiveJacobianSlice &slice :
             *active_jacobian_slices_)
        {
          if (!JacobianRequested(slice.block))
            continue;
          Jacobian(slice.block).block(
              slice.row_offset, slice.column_offset,
              slice.row_count, slice.column_count) *= scale;
        }
        return;
      }
      for (std::size_t block = 0;
           block < parameter_block_sizes_->size(); ++block)
      {
        if (JacobianRequested(block))
          Jacobian(block) *= scale;
      }
    }

  private:
    double *residuals_;
    double *const *jacobians_;
    const std::vector<int32_t> *parameter_block_sizes_;
    int row_offset_;
    int residual_dimension_;
    const std::vector<ActiveJacobianSlice> *active_jacobian_slices_;
  };

  class JacobianWriter
  {
  public:
    JacobianWriter(double *residuals, double *const *jacobians,
                   const int total_residual_count,
                   const std::vector<int32_t> &parameter_block_sizes,
                   const int observation_residual_dimension,
                   const JacobianWritePlan *plan = nullptr)
        : residuals_(residuals), jacobians_(jacobians),
          total_residual_count_(total_residual_count),
          parameter_block_sizes_(&parameter_block_sizes),
          observation_residual_dimension_(observation_residual_dimension),
          plan_(plan)
    {
      assert(residuals_ != nullptr);
      assert(total_residual_count_ > 0);
      assert(observation_residual_dimension_ > 0);
    }

    void ZeroRequestedJacobians() const
    {
      if (jacobians_ == nullptr)
        return;
      for (std::size_t block = 0;
           block < parameter_block_sizes_->size(); ++block)
      {
        if (jacobians_[block] == nullptr)
          continue;
        Eigen::Map<DynamicRowMajorMatrix> jacobian(
            jacobians_[block], total_residual_count_,
            (*parameter_block_sizes_)[block]);
        jacobian.setZero();
      }
    }

    void ZeroInactiveJacobians(const std::size_t observation_count) const
    {
      if (jacobians_ == nullptr)
        return;
      assert(plan_ != nullptr);
      for (const auto &slice : plan_->zero_slices)
      {
        if (jacobians_[slice.block] == nullptr)
          continue;
        Eigen::Map<DynamicRowMajorMatrix> block(
            jacobians_[slice.block], total_residual_count_,
            (*parameter_block_sizes_)[slice.block]);
        if (slice.row_offset == 0 && slice.row_count == observation_residual_dimension_)
        {
          block.block(0, slice.column_offset,
                      observation_count * observation_residual_dimension_,
                      slice.column_count).setZero();
          continue;
        }
        for (std::size_t i = 0; i < observation_count; ++i)
          block.block(i * observation_residual_dimension_ + slice.row_offset,
                      slice.column_offset, slice.row_count, slice.column_count).setZero();
      }
      const int data_rows = observation_count * observation_residual_dimension_;
      for (std::size_t b = 0; b < parameter_block_sizes_->size(); ++b)
        if (jacobians_[b] != nullptr && data_rows < total_residual_count_)
          Eigen::Map<DynamicRowMajorMatrix>(
              jacobians_[b], total_residual_count_, (*parameter_block_sizes_)[b])
              .bottomRows(total_residual_count_ - data_rows).setZero();
    }

    ObservationOutput Observation(const std::size_t index) const
    {
      const int row_offset =
          static_cast<int>(index) * observation_residual_dimension_;
      assert(row_offset + observation_residual_dimension_ <=
             total_residual_count_);
      return ObservationOutput(
          residuals_, jacobians_, *parameter_block_sizes_, row_offset,
          observation_residual_dimension_, plan_ ? &plan_->active_slices : nullptr);
    }

  private:
    double *residuals_;
    double *const *jacobians_;
    int total_residual_count_;
    const std::vector<int32_t> *parameter_block_sizes_;
    int observation_residual_dimension_;
    const JacobianWritePlan *plan_;
  };
} // namespace splbatch
