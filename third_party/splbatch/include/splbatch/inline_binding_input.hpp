#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "splbatch/batch_binding.hpp"

namespace splbatch
{
  namespace detail
  {
    // Metadata-only storage, deliberately not a replacement for public vector
    // types in BatchBinding. Copies own their arrays; no self-referential view
    // can dangle when an input is returned, copied or moved.
    template <typename T, std::size_t N>
    class InlineInputArray
    {
      static_assert(N > 0, "An input array needs positive inline capacity");
      static_assert(std::is_trivially_copyable<T>::value,
                    "Input arrays store trivial metadata only");
    public:
      InlineInputArray() = default;
      InlineInputArray(const InlineInputArray &other)
      {
        reserve(other.size());
        for (const auto value : other) push_back(value);
      }
      InlineInputArray(InlineInputArray &&other) noexcept
          : inline_(other.inline_), overflow_(std::move(other.overflow_)),
            size_(std::exchange(other.size_, 0)),
            spilled_(std::exchange(other.spilled_, false)) {}
      InlineInputArray &operator=(const InlineInputArray &other)
      {
        if (this != &other)
        {
          InlineInputArray copy(other);
          *this = std::move(copy);
        }
        return *this;
      }
      InlineInputArray &operator=(InlineInputArray &&other) noexcept
      {
        if (this != &other)
        {
          inline_ = other.inline_;
          overflow_ = std::move(other.overflow_);
          size_ = std::exchange(other.size_, 0);
          spilled_ = std::exchange(other.spilled_, false);
        }
        return *this;
      }
      std::size_t size() const noexcept { return size_; }
      std::size_t capacity() const noexcept { return spilled_ ? overflow_.capacity() : N; }
      bool empty() const noexcept { return size_ == 0; }
      bool IsInline() const noexcept { return !spilled_; }
      T *data() noexcept { return spilled_ ? overflow_.data() : inline_.data(); }
      const T *data() const noexcept { return spilled_ ? overflow_.data() : inline_.data(); }
      T *begin() noexcept { return data(); }
      const T *begin() const noexcept { return data(); }
      T *end() noexcept { return data() + size_; }
      const T *end() const noexcept { return data() + size_; }
      T &operator[](std::size_t i) noexcept { return data()[i]; }
      const T &operator[](std::size_t i) const noexcept { return data()[i]; }

      void reserve(std::size_t count)
      {
        if (count <= capacity()) return;
        const auto maximum = overflow_.max_size();
        if (count > maximum) throw std::length_error("Binding input array is too large");
        const auto grown = capacity() > maximum / 2 ? maximum : 2 * capacity();
        const auto requested = std::max(count, grown);
        if (spilled_)
          overflow_.reserve(requested);
        else
        {
          std::vector<T> staged;
          staged.reserve(requested);
          staged.assign(inline_.begin(), inline_.begin() + size_);
          overflow_.swap(staged);
          spilled_ = true;
        }
      }
      void push_back(T value)
      {
        if (size_ == overflow_.max_size()) throw std::length_error("Binding input array is full");
        reserve(size_ + 1);
        if (spilled_) overflow_.push_back(value);
        else inline_[size_] = value;
        ++size_;
      }
      void pop_back() noexcept
      {
        if (spilled_) overflow_.pop_back();
        --size_;
      }
      void clear() noexcept
      {
        overflow_.clear();
        size_ = 0;
      }
    private:
      std::array<T, N> inline_{};
      std::vector<T> overflow_;
      std::size_t size_ = 0;
      bool spilled_ = false;
    };
  } // namespace detail

  template <std::size_t N>
  struct InlineParameterLayout
  {
    detail::InlineInputArray<double *, N> blocks;
    detail::InlineInputArray<int32_t, N> sizes;

    void AddParameter(double *parameter, int32_t size)
    {
      // Reserve both before appending either. A spill failure can grow capacity
      // but cannot leave pointer and size counts inconsistent.
      blocks.reserve(blocks.size() + 1);
      sizes.reserve(sizes.size() + 1);
      blocks.push_back(parameter);
      sizes.push_back(size);
    }
    void Validate() const
    {
      // Same rules and error ordering as the preserved owning layout API.
      if (blocks.size() != sizes.size())
        throw std::invalid_argument("Parameter block pointer and size counts differ");
      for (std::size_t i = 0; i < blocks.size(); ++i)
      {
        if (blocks[i] == nullptr)
          throw std::invalid_argument("Parameter block pointer is null");
        if (sizes[i] <= 0)
          throw std::invalid_argument("Parameter block size is not positive");
        for (std::size_t j = 0; j < i; ++j)
          if (blocks[j] == blocks[i])
            throw std::invalid_argument(
                "A Ceres residual block cannot contain duplicate parameter block pointers");
      }
    }
  };

  // Owns its metadata, borrows only the actual optimization parameters. The
  // channel never retains references into this input. Prepared observations
  // must own any input metadata/context they need after PrepareObservation.
  template <typename ContextT, std::size_t InlineBlocks = 16,
            std::size_t InlineSignature = 4>
  struct InlineBindingInput
  {
    using Context = ContextT;
    Context context;
    InlineParameterLayout<InlineBlocks> parameters;
    detail::InlineInputArray<std::size_t, InlineSignature> support_signature;

    void AddParameter(double *parameter, int32_t size)
    { parameters.AddParameter(parameter, size); }

    // Explicit conversion at an ownership boundary, not used for lookup.
    // Validation belongs to channel ingestion (or the owning Binding's Key()).
    BatchBinding<Context> ToBinding() const
    {
      BatchBinding<Context> binding{context, {}, {}};
      binding.parameters.blocks.assign(parameters.blocks.begin(), parameters.blocks.end());
      binding.parameters.sizes.assign(parameters.sizes.begin(), parameters.sizes.end());
      binding.support_signature.assign(support_signature.begin(), support_signature.end());
      return binding;
    }
  };

  // Opt in per evaluator, never by guessed Context equality. Compatible means
  // preparing with the stored Binding has the same semantics as this input:
  // support was already compared completely, and all preparation context must
  // be covered here. Do not opt in for identity-sensitive/borrowed metadata.
  // The default uses a fresh owning Binding on every lightweight cache hit.
  template <typename Evaluator>
  struct BindingPreparationReuse
  {
    template <typename Context>
    static bool Compatible(const typename Evaluator::Binding &, const Context &)
    { return false; }
  };
} // namespace splbatch
