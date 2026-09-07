#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splbatch
{
  /// A half-open range of residual rows within one observation.
  struct ResidualRange
  {
    int32_t offset = 0;
    int32_t count = 0;
  };

  /// A structurally nonzero column range within one parameter block.
  struct ParameterBlockSlice
  {
    std::size_t block = 0;
    int32_t column_offset = 0;
    int32_t column_count = 0;
  };

  /// All parameter slices that can be nonzero for one residual-row range.
  /// Residual ranges in a JacobianStructure must form a disjoint partition
  /// of the evaluator's per-observation residual rows.
  struct JacobianGroup
  {
    ResidualRange residual_rows;
    std::vector<ParameterBlockSlice> parameter_slices;
  };

  struct JacobianStructure
  {
    std::vector<JacobianGroup> groups;
  };

  inline JacobianStructure MakeDenseJacobianStructure(
      const int residual_dimension,
      const std::vector<int32_t> &parameter_block_sizes)
  {
    if (residual_dimension <= 0)
      throw std::invalid_argument("Residual dimension must be positive");

    JacobianGroup group;
    group.residual_rows = {0, residual_dimension};
    group.parameter_slices.reserve(parameter_block_sizes.size());
    for (std::size_t block = 0;
         block < parameter_block_sizes.size(); ++block)
    {
      if (parameter_block_sizes[block] <= 0)
      {
        throw std::invalid_argument(
            "Parameter block size must be positive");
      }
      group.parameter_slices.push_back(
          {block, 0, parameter_block_sizes[block]});
    }
    return JacobianStructure{{std::move(group)}};
  }

  namespace detail
  {
    struct PlannedParameterSlice
    {
      std::size_t block = 0;
      int32_t source_column_offset = 0;
      int32_t column_count = 0;
      int32_t group_column_offset = 0;
    };

    struct PlannedJacobianGroup
    {
      ResidualRange residual_rows;
      int component = -1;
      int column_count = 0;
      std::vector<PlannedParameterSlice> parameter_slices;
      std::vector<int> component_columns;
    };

    struct PlannedActiveColumn
    {
      std::size_t block = 0;
      int ambient_column = 0;
    };

    /// Precompiled reverse lookup used while an evaluator writes one
    /// parameter block. It avoids rescanning every Jacobian group for each
    /// small fixed-size block assignment.
    struct PlannedBlockAccess
    {
      std::size_t group = 0;
      ResidualRange residual_rows;
      int32_t source_column_offset = 0;
      int32_t column_count = 0;
      int32_t group_column_offset = 0;
    };

    struct PlannedNormalComponent
    {
      int output_row_offset = 0;
      std::vector<PlannedActiveColumn> columns;
    };

    class DisjointSet
    {
    public:
      explicit DisjointSet(const int size)
          : parent_(static_cast<std::size_t>(size)),
            rank_(static_cast<std::size_t>(size), 0)
      {
        for (int index = 0; index < size; ++index)
          parent_[static_cast<std::size_t>(index)] = index;
      }

      int Find(const int value)
      {
        int &parent = parent_[static_cast<std::size_t>(value)];
        if (parent != value)
          parent = Find(parent);
        return parent;
      }

      void Union(const int lhs, const int rhs)
      {
        int lhs_root = Find(lhs);
        int rhs_root = Find(rhs);
        if (lhs_root == rhs_root)
          return;
        if (rank_[static_cast<std::size_t>(lhs_root)] <
            rank_[static_cast<std::size_t>(rhs_root)])
          std::swap(lhs_root, rhs_root);
        parent_[static_cast<std::size_t>(rhs_root)] = lhs_root;
        if (rank_[static_cast<std::size_t>(lhs_root)] ==
            rank_[static_cast<std::size_t>(rhs_root)])
        {
          ++rank_[static_cast<std::size_t>(lhs_root)];
        }
      }

    private:
      std::vector<int> parent_;
      std::vector<int> rank_;
    };

    /// Validated, immutable execution plan compiled once per batch factor.
    class CompiledJacobianStructure
    {
    public:
      CompiledJacobianStructure(
          const JacobianStructure &structure,
          const int residual_dimension,
          const std::vector<int32_t> &parameter_block_sizes)
      {
        if (residual_dimension <= 0)
          throw std::invalid_argument("Residual dimension must be positive");
        if (structure.groups.empty())
          throw std::invalid_argument("Jacobian structure has no groups");

        std::vector<int> row_owner(
            static_cast<std::size_t>(residual_dimension), -1);
        std::vector<std::vector<int>> active_ids_by_block;
        active_ids_by_block.reserve(parameter_block_sizes.size());
        for (const int32_t block_size : parameter_block_sizes)
        {
          if (block_size <= 0)
          {
            throw std::invalid_argument(
                "Parameter block size must be positive");
          }
          active_ids_by_block.emplace_back(
              static_cast<std::size_t>(block_size), -1);
        }

        std::vector<PlannedActiveColumn> active_columns;
        std::vector<std::vector<int>> group_active_ids;
        group_active_ids.reserve(structure.groups.size());
        for (std::size_t group_index = 0;
             group_index < structure.groups.size(); ++group_index)
        {
          const JacobianGroup &group = structure.groups[group_index];
          const int64_t residual_end =
              static_cast<int64_t>(group.residual_rows.offset) +
              group.residual_rows.count;
          if (group.residual_rows.offset < 0 ||
              group.residual_rows.count <= 0 ||
              residual_end > residual_dimension)
          {
            throw std::invalid_argument(
                "Jacobian group residual range is invalid");
          }
          for (int row = group.residual_rows.offset;
               row < residual_end; ++row)
          {
            int &owner = row_owner[static_cast<std::size_t>(row)];
            if (owner >= 0)
            {
              throw std::invalid_argument(
                  "Jacobian group residual ranges overlap");
            }
            owner = static_cast<int>(group_index);
          }

          std::vector<int> active_ids;
          for (const ParameterBlockSlice &slice : group.parameter_slices)
          {
            if (slice.block >= parameter_block_sizes.size())
            {
              throw std::invalid_argument(
                  "Jacobian parameter slice block is out of range");
            }
            const int64_t column_end =
                static_cast<int64_t>(slice.column_offset) +
                slice.column_count;
            if (slice.column_offset < 0 || slice.column_count <= 0 ||
                column_end > parameter_block_sizes[slice.block])
            {
              throw std::invalid_argument(
                  "Jacobian parameter slice column range is invalid");
            }
            for (int column = slice.column_offset;
                 column < column_end; ++column)
            {
              int &active_id = active_ids_by_block[slice.block]
                  [static_cast<std::size_t>(column)];
              if (active_id < 0)
              {
                active_id = static_cast<int>(active_columns.size());
                active_columns.push_back(
                    {slice.block, column});
              }
              for (const int previous : active_ids)
              {
                if (previous == active_id)
                {
                  throw std::invalid_argument(
                      "Jacobian group contains a duplicate parameter column");
                }
              }
              active_ids.push_back(active_id);
            }
          }
          group_active_ids.push_back(std::move(active_ids));
        }

        for (const int owner : row_owner)
        {
          if (owner < 0)
          {
            throw std::invalid_argument(
                "Jacobian groups do not cover every residual row");
          }
        }

        DisjointSet components(static_cast<int>(active_columns.size()));
        for (const std::vector<int> &active_ids : group_active_ids)
        {
          for (std::size_t index = 1; index < active_ids.size(); ++index)
            components.Union(active_ids[0], active_ids[index]);
        }

        std::vector<int> root_to_component(active_columns.size(), -1);
        std::vector<int> active_component(active_columns.size(), -1);
        std::vector<int> active_component_column(active_columns.size(), -1);
        for (std::size_t active_id = 0;
             active_id < active_columns.size(); ++active_id)
        {
          const int root = components.Find(static_cast<int>(active_id));
          int &component = root_to_component[static_cast<std::size_t>(root)];
          if (component < 0)
          {
            component = static_cast<int>(components_.size());
            components_.emplace_back();
          }
          active_component[active_id] = component;
          active_component_column[active_id] =
              static_cast<int>(components_[component].columns.size());
          components_[component].columns.push_back(active_columns[active_id]);
        }

        groups_.reserve(structure.groups.size());
        for (std::size_t group_index = 0;
             group_index < structure.groups.size(); ++group_index)
        {
          const JacobianGroup &source = structure.groups[group_index];
          const std::vector<int> &active_ids =
              group_active_ids[group_index];
          PlannedJacobianGroup group;
          group.residual_rows = source.residual_rows;
          if (!active_ids.empty())
            group.component = active_component[active_ids.front()];

          int group_column_offset = 0;
          std::size_t active_offset = 0;
          group.parameter_slices.reserve(source.parameter_slices.size());
          for (const ParameterBlockSlice &slice : source.parameter_slices)
          {
            group.parameter_slices.push_back(
                {slice.block, slice.column_offset, slice.column_count,
                 group_column_offset});
            for (int column = 0; column < slice.column_count; ++column)
            {
              const int active_id = active_ids[active_offset++];
              if (active_component[active_id] != group.component)
              {
                throw std::logic_error(
                    "A Jacobian group spans disconnected components");
              }
              group.component_columns.push_back(
                  active_component_column[active_id]);
            }
            group_column_offset += slice.column_count;
          }
          group.column_count = group_column_offset;
          groups_.push_back(std::move(group));
        }

        block_accesses_.resize(parameter_block_sizes.size());
        for (std::size_t group_index = 0;
             group_index < groups_.size(); ++group_index)
        {
          const PlannedJacobianGroup &group = groups_[group_index];
          for (const PlannedParameterSlice &slice :
               group.parameter_slices)
          {
            block_accesses_[slice.block].push_back(
                {group_index, group.residual_rows,
                 slice.source_column_offset, slice.column_count,
                 slice.group_column_offset});
          }
        }

        int output_row_offset = 0;
        for (PlannedNormalComponent &component : components_)
        {
          if (component.columns.size() >
              static_cast<std::size_t>(
                  std::numeric_limits<int>::max() - output_row_offset))
          {
            throw std::overflow_error(
                "Jacobian structure active dimension is too large");
          }
          component.output_row_offset = output_row_offset;
          output_row_offset += static_cast<int>(component.columns.size());
        }
        active_dimension_ = output_row_offset;
      }

      int ActiveDimension() const { return active_dimension_; }

      const std::vector<PlannedJacobianGroup> &Groups() const
      {
        return groups_;
      }

      const std::vector<PlannedNormalComponent> &Components() const
      {
        return components_;
      }

      const std::vector<std::vector<PlannedBlockAccess>> &
      BlockAccesses() const
      {
        return block_accesses_;
      }

    private:
      std::vector<PlannedJacobianGroup> groups_;
      std::vector<PlannedNormalComponent> components_;
      std::vector<std::vector<PlannedBlockAccess>> block_accesses_;
      int active_dimension_ = 0;
    };
  } // namespace detail
} // namespace splbatch
