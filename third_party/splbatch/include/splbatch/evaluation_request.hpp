#pragma once

#include <cstddef>

namespace splbatch
{
  class EvaluationRequest
  {
  public:
    EvaluationRequest(double *const *jacobians,
                      const std::size_t parameter_block_count)
        : jacobians_(jacobians),
          parameter_block_count_(parameter_block_count)
    {
    }

    bool JacobianRequested(const std::size_t block) const
    {
      return jacobians_ != nullptr && block < parameter_block_count_ &&
             jacobians_[block] != nullptr;
    }

    bool AnyJacobianRequested() const
    {
      if (jacobians_ == nullptr)
        return false;
      for (std::size_t block = 0; block < parameter_block_count_; ++block)
      {
        if (jacobians_[block] != nullptr)
          return true;
      }
      return false;
    }

  private:
    double *const *jacobians_;
    std::size_t parameter_block_count_;
  };
} // namespace splbatch
