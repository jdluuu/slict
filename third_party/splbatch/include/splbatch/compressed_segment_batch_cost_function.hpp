#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <ceres/cost_function.h>

#include "splbatch/evaluation_request.hpp"
#include "splbatch/evaluator_base.hpp"
#include "splbatch/jacobian_structure.hpp"
#include "splbatch/jacobian_writer.hpp"
#include "splbatch/loss_policy.hpp"
#include "splbatch/normal_equation_compressor.hpp"

namespace splbatch
{
  namespace detail
  {
    template <typename Evaluator, typename = void>
    struct StaticNormalGroupShape
    {
      static constexpr int kRows = Eigen::Dynamic;
      static constexpr int kColumns = Eigen::Dynamic;
      static constexpr bool kEnabled = false;
    };

    /// Optional evaluator fast-path contract. An evaluator may declare
    /// kStaticNormalGroupRows and kStaticNormalGroupColumns when every
    /// Jacobian group has the same compile-time shape. The generic compressor
    /// can then use fixed-size Eigen kernels for direct per-observation H/g
    /// accumulation while retaining the dynamic batched fallback otherwise.
    template <typename Evaluator>
    struct StaticNormalGroupShape<
        Evaluator,
        std::void_t<
            decltype(Evaluator::kStaticNormalGroupRows),
            decltype(Evaluator::kStaticNormalGroupColumns)>>
    {
      static constexpr int kRows =
          Evaluator::kStaticNormalGroupRows;
      static constexpr int kColumns =
          Evaluator::kStaticNormalGroupColumns;
      static constexpr bool kEnabled = kRows > 0 && kColumns > 0;
    };

    template <int Rows, int Columns>
    class FixedGroupNormalAccumulator
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      using LocalJacobian =
          Eigen::Matrix<double, Rows, Columns, Eigen::RowMajor>;
      using HessianMatrix =
          Eigen::Matrix<double, Columns, Columns>;
      using GradientVector =
          Eigen::Matrix<double, Columns, 1>;

      FixedGroupNormalAccumulator(const int, const int rows,
                                  const int columns)
      {
        if (rows != Rows || columns != Columns)
        {
          throw std::logic_error(
              "Evaluator static normal-group shape does not match its "
              "JacobianStructure");
        }
        local_jacobian_.setZero();
        hessian_.setZero();
        gradient_.setZero();
      }

      template <typename Derived>
      void SetSlice(const std::size_t, const int destination_column,
                    const Eigen::MatrixBase<Derived> &source)
      {
        local_jacobian_.block(
            0, destination_column, Rows, source.cols()) = source;
      }

      void FinishObservation(const std::size_t, const double *residual)
      {
        const Eigen::Map<const Eigen::Matrix<double, Rows, 1>>
            residual_map(residual);
        hessian_.noalias() +=
            local_jacobian_.transpose() * local_jacobian_;
        gradient_.noalias() +=
            local_jacobian_.transpose() * residual_map;
      }

      void Finalize() {}

      LocalJacobian &Jacobian() { return local_jacobian_; }
      const LocalJacobian &Jacobian() const { return local_jacobian_; }

      double Hessian(const int row, const int column) const
      {
        return hessian_(row, column);
      }

      double Gradient(const int row) const { return gradient_[row]; }

      const HessianMatrix &HessianValue() const { return hessian_; }
      const GradientVector &GradientValue() const { return gradient_; }

    private:
      LocalJacobian local_jacobian_;
      HessianMatrix hessian_;
      GradientVector gradient_;
    };

    class BatchedGroupNormalAccumulator
    {
    public:
      BatchedGroupNormalAccumulator(const int observation_count,
                                    const int rows, const int columns)
          : rows_(rows), columns_(columns),
            jacobian_(observation_count * rows, columns),
            residual_(observation_count * rows),
            hessian_(columns, columns), gradient_(columns)
      {
      }

      template <typename Derived>
      void SetSlice(const std::size_t observation_index,
                    const int destination_column,
                    const Eigen::MatrixBase<Derived> &source)
      {
        jacobian_.block(
            static_cast<int>(observation_index) * rows_,
            destination_column, rows_, source.cols()) = source;
      }

      void FinishObservation(const std::size_t observation_index,
                             const double *residual)
      {
        residual_.segment(
            static_cast<int>(observation_index) * rows_, rows_) =
            Eigen::Map<const Eigen::VectorXd>(residual, rows_);
      }

      void Finalize()
      {
        hessian_.noalias() = jacobian_.transpose() * jacobian_;
        gradient_.noalias() = jacobian_.transpose() * residual_;
      }

      double Hessian(const int row, const int column) const
      {
        return hessian_(row, column);
      }

      double Gradient(const int row) const { return gradient_[row]; }

    private:
      int rows_;
      int columns_;
      DynamicRowMajorMatrix jacobian_;
      Eigen::VectorXd residual_;
      Eigen::MatrixXd hessian_;
      Eigen::VectorXd gradient_;
    };

    template <typename GroupAccumulator>
    class StructuredObservationOutput : public ResidualProjection
    {
    public:
      using GroupJacobian = typename GroupAccumulator::LocalJacobian;

      class JacobianBlock
      {
      public:
        JacobianBlock(
            const std::size_t block,
            const CompiledJacobianStructure &structure,
            std::vector<GroupAccumulator> &accumulators)
            : block_(block), structure_(&structure),
              accumulators_(&accumulators)
        {
        }

        template <int Rows, int Columns>
        auto block(const int row_offset, const int column_offset) const
        {
          if (block_ >= structure_->BlockAccesses().size())
          {
            throw std::out_of_range(
                "Jacobian parameter block is out of range");
          }
          for (const PlannedBlockAccess &access :
               structure_->BlockAccesses()[block_])
          {
            const int group_row_offset =
                row_offset - access.residual_rows.offset;
            if (group_row_offset < 0 ||
                group_row_offset + Rows > access.residual_rows.count)
            {
              continue;
            }
            if (column_offset < access.source_column_offset ||
                column_offset + Columns >
                    access.source_column_offset + access.column_count)
            {
              continue;
            }
            const int group_column_offset =
                access.group_column_offset + column_offset -
                access.source_column_offset;
            return (*accumulators_)[access.group]
                .Jacobian()
                .template block<Rows, Columns>(
                    group_row_offset, group_column_offset);
          }
          throw std::out_of_range(
              "Requested Jacobian block is outside the declared structure");
        }

      private:
        std::size_t block_;
        const CompiledJacobianStructure *structure_;
        std::vector<GroupAccumulator> *accumulators_;
      };

      StructuredObservationOutput(
          double *residuals,
          const std::vector<int32_t> &parameter_block_sizes,
          const int residual_dimension,
          const CompiledJacobianStructure &structure,
          std::vector<GroupAccumulator> &accumulators)
          : residuals_(residuals),
            parameter_block_sizes_(&parameter_block_sizes),
            residual_dimension_(residual_dimension),
            structure_(&structure), accumulators_(&accumulators)
      {
      }

      Eigen::Map<Eigen::VectorXd> Residual() const
      {
        return Eigen::Map<Eigen::VectorXd>(
            residuals_, residual_dimension_);
      }

      bool JacobianRequested(const std::size_t block) const
      {
        return block < structure_->BlockAccesses().size() &&
               !structure_->BlockAccesses()[block].empty();
      }

      JacobianBlock Jacobian(const std::size_t block) const
      {
        if (!JacobianRequested(block))
        {
          throw std::out_of_range(
              "Jacobian block is not part of the declared structure");
        }
        return JacobianBlock(
            block, *structure_, *accumulators_);
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
          residuals_[row] *= scale;
        for (GroupAccumulator &accumulator : *accumulators_)
          accumulator.Jacobian() *= scale;
      }

    private:
      double *residuals_;
      const std::vector<int32_t> *parameter_block_sizes_;
      int residual_dimension_;
      const CompiledJacobianStructure *structure_;
      std::vector<GroupAccumulator> *accumulators_;
    };
  } // namespace detail

  /// Condenses all observations in a batch to an equivalent Gauss-Newton
  /// model. At every Evaluate(), local H = sum(J^T J),
  /// g = sum(J^T r), and the true robust cost are rebuilt from current
  /// parameters. The synthetic residual [b; slack] and Jacobian [A; 0]
  /// preserve cost, gradient, and Gauss-Newton Hessian with
  /// A^T A = H and A^T b = g. If the evaluator declares its structurally
  /// nonzero Jacobian slices, disconnected normal-equation components are
  /// assembled and factorized independently. Evaluators without a declaration
  /// automatically use one conservative dense component.
  template <typename Evaluator, typename LossPolicy>
  class CompressedSegmentBatchCostFunction final
      : public ceres::CostFunction
  {
    static_assert(
        kIsEvaluator<Evaluator>,
        "Evaluator must derive from splbatch::EvaluatorBase");

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Binding = typename Evaluator::Binding;
    using PreparedObservation = typename Evaluator::PreparedObservation;
    using SharedPreparedObservations =
        std::shared_ptr<std::vector<PreparedObservation>>;
    using Runtime = typename Evaluator::Runtime;

    CompressedSegmentBatchCostFunction(
        std::shared_ptr<const Evaluator> evaluator, Binding binding,
        std::vector<PreparedObservation> observations,
        LossPolicy loss_policy)
        : CompressedSegmentBatchCostFunction(
              std::move(evaluator), std::move(binding),
              std::move(observations), {}, std::move(loss_policy))
    {
    }

    CompressedSegmentBatchCostFunction(
        std::shared_ptr<const Evaluator> evaluator, Binding binding,
        std::vector<PreparedObservation> observations,
        std::vector<double> observation_loss_scales,
        LossPolicy loss_policy)
        : evaluator_(std::move(evaluator)), binding_(std::move(binding)),
          observations_(std::move(observations)),
          observation_loss_scales_(std::move(observation_loss_scales)),
          loss_policy_(std::move(loss_policy))
    {
      Initialize();
    }

    /// Incremental registration shares one contiguous observation array so
    /// coefficient refreshes stay visible without per-row pointer chasing.
    CompressedSegmentBatchCostFunction(
        std::shared_ptr<const Evaluator> evaluator, Binding binding,
        SharedPreparedObservations observations,
        std::vector<double> observation_loss_scales,
        LossPolicy loss_policy)
        : evaluator_(std::move(evaluator)), binding_(std::move(binding)),
          shared_observations_(std::move(observations)),
          observation_loss_scales_(std::move(observation_loss_scales)),
          loss_policy_(std::move(loss_policy))
    {
      Initialize();
    }

    bool Evaluate(double const *const *parameters, double *residuals,
                  double **jacobians) const override
    {
      Eigen::Map<Eigen::VectorXd> compact_output(
          residuals, num_residuals());
      compact_output.setZero();

      bool need_jacobians = false;
      if (jacobians != nullptr)
      {
        for (std::size_t block = 0;
             block < binding_.parameters.sizes.size(); ++block)
        {
          need_jacobians |= jacobians[block] != nullptr;
        }
      }

      if (!need_jacobians)
      {
        Eigen::Matrix<double, Evaluator::kResidualDim, 1>
            observation_residual;
        const EvaluationRequest request(
            nullptr, binding_.parameters.sizes.size());
        const Runtime runtime =
            evaluator_->PrepareBatch(binding_, parameters, request);
        double squared_residual_norm = 0.0;
        for (std::size_t observation_index = 0;
             observation_index < ObservationCount();
             ++observation_index)
        {
          observation_residual.setZero();
          const ObservationOutput output(
              observation_residual.data(), nullptr,
              binding_.parameters.sizes, 0, Evaluator::kResidualDim);
          if (!evaluator_->EvaluateObservation(
                  binding_, ObservationAt(observation_index), runtime,
                  parameters, output))
          {
            return false;
          }

          const double cost_gap = loss_policy_.Apply(
              output, observation_loss_scales_[observation_index]);
          squared_residual_norm +=
              observation_residual.squaredNorm() + cost_gap;
          if (!std::isfinite(squared_residual_norm))
            return false;
        }
        compact_output[active_dimension_] =
            std::sqrt(std::max(0.0, squared_residual_norm));
        return true;
      }

      Eigen::Matrix<double, Evaluator::kResidualDim, 1>
          observation_residual;
      using StaticGroupShape =
          detail::StaticNormalGroupShape<Evaluator>;
      using GroupNormalAccumulator = std::conditional_t<
          StaticGroupShape::kEnabled,
          detail::FixedGroupNormalAccumulator<
              StaticGroupShape::kRows, StaticGroupShape::kColumns>,
          detail::BatchedGroupNormalAccumulator>;

      std::vector<GroupNormalAccumulator> group_accumulators;
      group_accumulators.reserve(structure_plan_->Groups().size());
      for (const detail::PlannedJacobianGroup &group :
           structure_plan_->Groups())
      {
        group_accumulators.emplace_back(
            static_cast<int>(ObservationCount()),
            group.residual_rows.count, group.column_count);
      }

      double squared_residual_norm = 0.0;
      if constexpr (StaticGroupShape::kEnabled)
      {
        std::vector<double *> requested_blocks(
            binding_.parameters.sizes.size(), nullptr);
        for (const ActiveJacobianSlice &slice : active_jacobian_slices_)
          requested_blocks[slice.block] = observation_residual.data();
        const EvaluationRequest request(
            requested_blocks.data(), requested_blocks.size());
        const Runtime runtime =
            evaluator_->PrepareBatch(binding_, parameters, request);

        for (std::size_t observation_index = 0;
             observation_index < ObservationCount(); ++observation_index)
        {
          observation_residual.setZero();
          for (GroupNormalAccumulator &accumulator : group_accumulators)
            accumulator.Jacobian().setZero();

          const detail::StructuredObservationOutput<GroupNormalAccumulator>
              output(
                  observation_residual.data(), binding_.parameters.sizes,
                  Evaluator::kResidualDim, *structure_plan_,
                  group_accumulators);
          if (!evaluator_->EvaluateObservation(
                  binding_, ObservationAt(observation_index), runtime,
                  parameters, output))
          {
            return false;
          }

          const double cost_gap = loss_policy_.Apply(
              output, observation_loss_scales_[observation_index]);
          squared_residual_norm +=
              observation_residual.squaredNorm() + cost_gap;
          if (!std::isfinite(squared_residual_norm))
            return false;

          for (std::size_t group_index = 0;
               group_index < structure_plan_->Groups().size(); ++group_index)
          {
            const detail::PlannedJacobianGroup &group =
                structure_plan_->Groups()[group_index];
            if (group.column_count == 0)
              continue;
            group_accumulators[group_index].FinishObservation(
                observation_index,
                observation_residual.data() +
                    group.residual_rows.offset);
          }
        }
      }
      else
      {
        std::vector<DynamicRowMajorMatrix> observation_block_jacobians(
            binding_.parameters.sizes.size());
        std::vector<double *> block_jacobian_pointers(
            binding_.parameters.sizes.size(), nullptr);
        for (const detail::PlannedJacobianGroup &group :
             structure_plan_->Groups())
        {
          for (const detail::PlannedParameterSlice &slice :
               group.parameter_slices)
          {
            if (block_jacobian_pointers[slice.block] != nullptr)
              continue;
            observation_block_jacobians[slice.block] =
                DynamicRowMajorMatrix::Zero(
                    Evaluator::kResidualDim,
                    binding_.parameters.sizes[slice.block]);
            block_jacobian_pointers[slice.block] =
                observation_block_jacobians[slice.block].data();
          }
        }
        const EvaluationRequest request(
            block_jacobian_pointers.data(),
            binding_.parameters.sizes.size());
        const Runtime runtime =
            evaluator_->PrepareBatch(binding_, parameters, request);

        for (std::size_t observation_index = 0;
             observation_index < ObservationCount(); ++observation_index)
        {
          observation_residual.setZero();
          for (std::size_t block = 0;
               block < block_jacobian_pointers.size(); ++block)
          {
            if (block_jacobian_pointers[block] != nullptr)
              observation_block_jacobians[block].setZero();
          }

          const ObservationOutput output(
              observation_residual.data(), block_jacobian_pointers.data(),
              binding_.parameters.sizes, 0, Evaluator::kResidualDim,
              &active_jacobian_slices_);
          if (!evaluator_->EvaluateObservation(
                  binding_, ObservationAt(observation_index), runtime,
                  parameters, output))
          {
            return false;
          }

          const double cost_gap = loss_policy_.Apply(
              output, observation_loss_scales_[observation_index]);
          squared_residual_norm +=
              observation_residual.squaredNorm() + cost_gap;
          if (!std::isfinite(squared_residual_norm))
            return false;

          for (std::size_t group_index = 0;
               group_index < structure_plan_->Groups().size(); ++group_index)
          {
            const detail::PlannedJacobianGroup &group =
                structure_plan_->Groups()[group_index];
            if (group.column_count == 0)
              continue;
            GroupNormalAccumulator &accumulator =
                group_accumulators[group_index];
            for (const detail::PlannedParameterSlice &slice :
                 group.parameter_slices)
            {
              accumulator.SetSlice(
                  observation_index, slice.group_column_offset,
                  observation_block_jacobians[slice.block].block(
                      group.residual_rows.offset,
                      slice.source_column_offset,
                      group.residual_rows.count, slice.column_count));
            }
            accumulator.FinishObservation(
                observation_index,
                observation_residual.data() +
                    group.residual_rows.offset);
          }
        }
      }

      for (GroupNormalAccumulator &accumulator : group_accumulators)
        accumulator.Finalize();

      for (std::size_t block = 0;
           block < binding_.parameters.sizes.size(); ++block)
      {
        if (jacobians[block] == nullptr)
          continue;
        Eigen::Map<DynamicRowMajorMatrix> output_jacobian(
            jacobians[block], num_residuals(),
            binding_.parameters.sizes[block]);
        output_jacobian.setZero();
      }

      // Fast path for independent fixed-size groups. This is the same local
      // algebra as a hand-written compact factor: each observation directly
      // accumulates a fixed H/g, followed by one fixed-size factorization per
      // component. The generic compiled structure still decides where the
      // resulting columns are written in Ceres' ambient parameter blocks.
      if constexpr (StaticGroupShape::kEnabled)
      {
        if (use_static_component_fast_path_)
        {
          constexpr int kDimension = StaticGroupShape::kColumns;
          double projected_squared_norm = 0.0;
          for (std::size_t component_index = 0;
               component_index < structure_plan_->Components().size();
               ++component_index)
          {
            const detail::PlannedNormalComponent &component =
                structure_plan_->Components()[component_index];
            const GroupNormalAccumulator &accumulator =
                group_accumulators[static_cast<std::size_t>(
                    static_component_group_indices_[component_index])];
            Eigen::Matrix<double, kDimension, kDimension, Eigen::RowMajor>
                square_root;
            Eigen::Matrix<double, kDimension, 1> compressed_residual;
            if (!MakeFixedSquareRootNormalEquation<kDimension>(
                    accumulator.HessianValue(),
                    accumulator.GradientValue(), square_root,
                    compressed_residual))
            {
              return false;
            }

            compact_output.template segment<kDimension>(
                component.output_row_offset) = compressed_residual;
            projected_squared_norm += compressed_residual.squaredNorm();

            for (int column = 0; column < kDimension; ++column)
            {
              const detail::PlannedActiveColumn &active_column =
                  component.columns[static_cast<std::size_t>(column)];
              if (jacobians[active_column.block] == nullptr)
                continue;
              Eigen::Map<DynamicRowMajorMatrix> output_jacobian(
                  jacobians[active_column.block], num_residuals(),
                  binding_.parameters.sizes[active_column.block]);
              output_jacobian.template block<kDimension, 1>(
                  component.output_row_offset,
                  active_column.ambient_column) = square_root.col(column);
            }
          }

          double slack_squared_norm =
              squared_residual_norm - projected_squared_norm;
          const double slack_tolerance = 1e-10 * std::max(
              {1.0, squared_residual_norm, projected_squared_norm});
          if (slack_squared_norm < -slack_tolerance)
            return false;
          compact_output[active_dimension_] =
              std::sqrt(std::max(0.0, slack_squared_norm));
          return true;
        }
      }

      struct ComponentNormalModel
      {
        explicit ComponentNormalModel(const int dimension)
            : hessian(Eigen::MatrixXd::Zero(dimension, dimension)),
              gradient(Eigen::VectorXd::Zero(dimension))
        {
        }

        Eigen::MatrixXd hessian;
        Eigen::VectorXd gradient;
      };

      std::vector<ComponentNormalModel> component_models;
      component_models.reserve(structure_plan_->Components().size());
      for (const detail::PlannedNormalComponent &component :
           structure_plan_->Components())
      {
        component_models.emplace_back(
            static_cast<int>(component.columns.size()));
      }

      // Scatter each independently accumulated residual-row group into its
      // connected normal-equation component. Fixed-shape evaluators reach
      // this point through direct local H/g accumulation; arbitrary dynamic
      // shapes use one compact, cache-friendly batch product per group.
      for (std::size_t group_index = 0;
           group_index < structure_plan_->Groups().size(); ++group_index)
      {
        const detail::PlannedJacobianGroup &group =
            structure_plan_->Groups()[group_index];
        if (group.column_count == 0)
          continue;
        const GroupNormalAccumulator &accumulator =
            group_accumulators[group_index];

        if (group.component < 0 ||
            static_cast<std::size_t>(group.component) >=
                component_models.size())
        {
          return false;
        }
        ComponentNormalModel &component =
            component_models[static_cast<std::size_t>(group.component)];
        for (int row = 0; row < group.column_count; ++row)
        {
          const int component_row = group.component_columns[row];
          component.gradient[component_row] += accumulator.Gradient(row);
          for (int column = 0; column < group.column_count; ++column)
          {
            component.hessian(
                component_row, group.component_columns[column]) +=
                accumulator.Hessian(row, column);
          }
        }
      }

      double projected_squared_norm = 0.0;
      for (std::size_t component_index = 0;
           component_index < structure_plan_->Components().size();
           ++component_index)
      {
        const detail::PlannedNormalComponent &component =
            structure_plan_->Components()[component_index];
        const ComponentNormalModel &model =
            component_models[component_index];
        const int component_dimension =
            static_cast<int>(component.columns.size());
        DynamicRowMajorMatrix square_root;
        Eigen::VectorXd compressed_residual;
        if (!MakeSquareRootNormalEquation(
                model.hessian, model.gradient,
                square_root, compressed_residual))
        {
          return false;
        }

        compact_output.segment(
            component.output_row_offset, component_dimension) =
            compressed_residual;
        projected_squared_norm += compressed_residual.squaredNorm();

        for (int column = 0; column < component_dimension; ++column)
        {
          const detail::PlannedActiveColumn &active_column =
              component.columns[static_cast<std::size_t>(column)];
          if (jacobians[active_column.block] == nullptr)
            continue;
          Eigen::Map<DynamicRowMajorMatrix> output_jacobian(
              jacobians[active_column.block], num_residuals(),
              binding_.parameters.sizes[active_column.block]);
          output_jacobian.block(
              component.output_row_offset,
              active_column.ambient_column,
              component_dimension, 1) = square_root.col(column);
        }
      }

      double slack_squared_norm =
          squared_residual_norm - projected_squared_norm;
      const double slack_tolerance = 1e-10 * std::max(
          {1.0, squared_residual_norm, projected_squared_norm});
      if (slack_squared_norm < -slack_tolerance)
        return false;
      slack_squared_norm = std::max(0.0, slack_squared_norm);
      compact_output[active_dimension_] =
          std::sqrt(slack_squared_norm);
      return true;
    }

  private:
    void Initialize()
    {
      if (!evaluator_)
        throw std::invalid_argument("Batch evaluator is null");
      if (!observations_.empty() && shared_observations_)
      {
        throw std::invalid_argument(
            "A compressed batch cannot mix owned and shared observations");
      }
      if (ObservationCount() == 0)
        throw std::invalid_argument("A compressed batch cannot be empty");
      if (observation_loss_scales_.empty())
        observation_loss_scales_.assign(ObservationCount(), 1.0);
      if (observation_loss_scales_.size() != ObservationCount())
      {
        throw std::invalid_argument(
            "Observation and loss-scale counts differ");
      }
      for (const double loss_scale : observation_loss_scales_)
        ValidateObservationLossScale(loss_scale);

      binding_.parameters.Validate();
      if (binding_.parameters.sizes.empty())
      {
        throw std::invalid_argument(
            "A compressed batch requires parameter blocks");
      }
      for (const int32_t size : binding_.parameters.sizes)
        mutable_parameter_block_sizes()->push_back(size);

      structure_plan_ =
          std::make_unique<const detail::CompiledJacobianStructure>(
              evaluator_->GetJacobianStructure(binding_),
              Evaluator::kResidualDim, binding_.parameters.sizes);
      for (const detail::PlannedJacobianGroup &group :
           structure_plan_->Groups())
      {
        for (const detail::PlannedParameterSlice &slice :
             group.parameter_slices)
        {
          active_jacobian_slices_.push_back(
              ActiveJacobianSlice{
                  slice.block, group.residual_rows.offset,
                  group.residual_rows.count,
                  slice.source_column_offset, slice.column_count});
        }
      }

      // A common spline layout has one independent fixed-size Jacobian group
      // per normal-equation component (for pose: one 3x12 rotation group and
      // one 3x12 position group). Remember that one-to-one mapping once so
      // Evaluate() can factor each group's fixed H/g directly, without first
      // scattering it through dynamic component matrices.
      if constexpr (detail::StaticNormalGroupShape<Evaluator>::kEnabled)
      {
        using StaticGroupShape =
            detail::StaticNormalGroupShape<Evaluator>;
        const auto &groups = structure_plan_->Groups();
        const auto &components = structure_plan_->Components();
        static_component_group_indices_.assign(components.size(), -1);
        use_static_component_fast_path_ =
            groups.size() == components.size();
        for (std::size_t group_index = 0;
             use_static_component_fast_path_ &&
             group_index < groups.size(); ++group_index)
        {
          const detail::PlannedJacobianGroup &group = groups[group_index];
          if (group.residual_rows.count != StaticGroupShape::kRows ||
              group.column_count != StaticGroupShape::kColumns ||
              group.component < 0 ||
              static_cast<std::size_t>(group.component) >=
                  components.size() ||
              components[static_cast<std::size_t>(group.component)]
                      .columns.size() !=
                  static_cast<std::size_t>(StaticGroupShape::kColumns) ||
              group.component_columns.size() !=
                  static_cast<std::size_t>(StaticGroupShape::kColumns) ||
              static_component_group_indices_[
                  static_cast<std::size_t>(group.component)] >= 0)
          {
            use_static_component_fast_path_ = false;
            break;
          }
          for (int column = 0;
               column < StaticGroupShape::kColumns; ++column)
          {
            if (group.component_columns[static_cast<std::size_t>(column)] !=
                column)
            {
              use_static_component_fast_path_ = false;
              break;
            }
          }
          if (use_static_component_fast_path_)
          {
            static_component_group_indices_[
                static_cast<std::size_t>(group.component)] =
                static_cast<int>(group_index);
          }
        }
        for (const int group_index : static_component_group_indices_)
          use_static_component_fast_path_ &= group_index >= 0;
      }
      active_dimension_ = structure_plan_->ActiveDimension();
      if (ObservationCount() >
          static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                   Evaluator::kResidualDim))
      {
        throw std::overflow_error(
            "Compressed batch residual dimension is too large");
      }
      set_num_residuals(active_dimension_ + 1);
    }

    std::size_t ObservationCount() const
    {
      return shared_observations_
                 ? shared_observations_->size()
                 : observations_.size();
    }

    const PreparedObservation &ObservationAt(
        const std::size_t index) const
    {
      if (!shared_observations_)
        return observations_.at(index);
      return shared_observations_->at(index);
    }

    std::shared_ptr<const Evaluator> evaluator_;
    Binding binding_;
    std::vector<PreparedObservation> observations_;
    SharedPreparedObservations shared_observations_;
    std::vector<double> observation_loss_scales_;
    LossPolicy loss_policy_;
    std::unique_ptr<const detail::CompiledJacobianStructure>
        structure_plan_;
    std::vector<ActiveJacobianSlice> active_jacobian_slices_;
    std::vector<int> static_component_group_indices_;
    bool use_static_component_fast_path_ = false;
    int active_dimension_ = 0;
  };
} // namespace splbatch
