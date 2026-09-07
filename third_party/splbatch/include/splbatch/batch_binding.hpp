#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splbatch
{
  struct ParameterBlockLayout
  {
    std::vector<double *> blocks;
    std::vector<int32_t> sizes;

    void Validate() const
    {
      if (blocks.size() != sizes.size())
      {
        throw std::invalid_argument(
            "Parameter block pointer and size counts differ");
      }
      for (std::size_t i = 0; i < blocks.size(); ++i)
      {
        if (blocks[i] == nullptr)
          throw std::invalid_argument("Parameter block pointer is null");
        if (sizes[i] <= 0)
          throw std::invalid_argument("Parameter block size is not positive");
        for (std::size_t previous = 0; previous < i; ++previous)
        {
          if (blocks[previous] == blocks[i])
          {
            throw std::invalid_argument(
                "A Ceres residual block cannot contain duplicate parameter "
                "block pointers");
          }
        }
      }
    }

    bool operator==(const ParameterBlockLayout &other) const
    {
      return blocks == other.blocks && sizes == other.sizes;
    }
  };

  inline std::size_t AppendUniqueParameterBlock(
      ParameterBlockLayout &layout, double *parameter, const int32_t size)
  {
    if (parameter == nullptr)
      throw std::invalid_argument("Parameter block pointer is null");
    if (size <= 0)
      throw std::invalid_argument("Parameter block size is not positive");

    for (std::size_t i = 0; i < layout.blocks.size(); ++i)
    {
      if (layout.blocks[i] != parameter)
        continue;
      if (layout.sizes[i] != size)
      {
        throw std::invalid_argument(
            "The same parameter pointer was registered with two sizes");
      }
      return i;
    }

    layout.blocks.push_back(parameter);
    layout.sizes.push_back(size);
    return layout.blocks.size() - 1;
  }

  struct BatchKey
  {
    std::vector<std::size_t> support_signature;
    std::vector<const double *> parameter_blocks;
    std::vector<int32_t> parameter_block_sizes;

    bool operator==(const BatchKey &other) const
    {
      return support_signature == other.support_signature &&
             parameter_blocks == other.parameter_blocks &&
             parameter_block_sizes == other.parameter_block_sizes;
    }
  };

  struct BatchKeyHash
  {
    std::size_t operator()(const BatchKey &key) const
    {
      std::size_t seed = 0;
      const auto combine = [&seed](const std::size_t value)
      {
        seed ^= value + static_cast<std::size_t>(0x9e3779b9) +
                (seed << 6) + (seed >> 2);
      };

      for (const std::size_t support : key.support_signature)
        combine(std::hash<std::size_t>{}(support));
      for (const double *parameter : key.parameter_blocks)
        combine(std::hash<const double *>{}(parameter));
      for (const int32_t size : key.parameter_block_sizes)
        combine(std::hash<int32_t>{}(size));
      return seed;
    }
  };

  namespace detail
  {
    // Non-owning lookup helpers for ordinary channel ingestion. These do not
    // cache anything in the caller's Binding or change the public Key() path
    // used by incremental channels. Context is deliberately not a support key.
    template <typename LeftBinding, typename RightBinding>
    bool SameBatchSupport(const LeftBinding &left, const RightBinding &right)
    {
      // Check every length before reading elements; a malformed prefix must
      // never hit a previously validated support and bypass validation.
      return left.parameters.blocks.size() == right.parameters.blocks.size() &&
             left.parameters.sizes.size() == right.parameters.sizes.size() &&
             left.support_signature.size() == right.support_signature.size() &&
             std::equal(left.parameters.blocks.begin(), left.parameters.blocks.end(),
                        right.parameters.blocks.begin()) &&
             std::equal(left.parameters.sizes.begin(), left.parameters.sizes.end(),
                        right.parameters.sizes.begin()) &&
             std::equal(left.support_signature.begin(), left.support_signature.end(),
                        right.support_signature.begin());
    }

    template <typename Binding>
    std::size_t HashBatchSupport(const Binding &binding)
    {
      // Same hash as BatchKeyHash, without constructing its owning vectors.
      // Hash collisions (including different array lengths) are resolved by
      // SameBatchSupport. Iterate arrays separately, even on invalid input.
      std::size_t seed = 0;
      const auto combine = [&seed](const std::size_t value)
      {
        seed ^= value + static_cast<std::size_t>(0x9e3779b9) +
                (seed << 6) + (seed >> 2);
      };
      for (const std::size_t support : binding.support_signature)
        combine(std::hash<std::size_t>{}(support));
      for (const double *parameter : binding.parameters.blocks)
        combine(std::hash<const double *>{}(parameter));
      for (const int32_t size : binding.parameters.sizes)
        combine(std::hash<int32_t>{}(size));
      return seed;
    }
  } // namespace detail

  template <typename ContextT>
  struct BatchBinding
  {
    using Context = ContextT;

    Context context;
    ParameterBlockLayout parameters;
    std::vector<std::size_t> support_signature;

    BatchKey Key() const
    {
      parameters.Validate();
      BatchKey key;
      key.support_signature = support_signature;
      key.parameter_blocks.reserve(parameters.blocks.size());
      for (const double *parameter : parameters.blocks)
        key.parameter_blocks.push_back(parameter);
      key.parameter_block_sizes = parameters.sizes;
      return key;
    }
  };
} // namespace splbatch
