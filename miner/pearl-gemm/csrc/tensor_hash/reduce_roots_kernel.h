#include "blake3/blake3.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "merkle_tree_utils.hpp"
#include "tensor_hash_constants.cuh"

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>

namespace pearl {

using namespace cute;

template <int kNumThreads>
class ReduceRootsKernel {
 public:
  using Element = uint32_t;
  static constexpr uint32_t MaxThreadsPerBlock = kNumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int SharedStorageSize =
      kNumThreads * blake3::CHAINING_VALUE_SIZE;

  struct Arguments {
    Element* ptr_roots;
    const uint32_t num_leaves;  // how many leaves do we reduce?
    const uint8_t* ptr_key;     // 32-byte BLAKE3 key in GMEM
    uint8_t* ptr_out;           // Final digest
  };

  struct Params {
    Element* ptr_roots;
    const uint32_t num_leaves;  // how many leaves do we reduce?
    const uint8_t* ptr_key;
    uint8_t* ptr_out;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{args.ptr_roots, args.num_leaves, args.ptr_key, args.ptr_out};
  }

  static dim3 get_grid_shape(Params const& params) { return dim3(1); }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    const int tid = threadIdx.x;
    // Copy each of the leaves from gmem -> smem
    // GMEM is structured as follows; we view it as an (8, num_leaves) matrix with 4-byte elements
    // Each leaf's 8 words are stored contiguously, leaves are spaced 8 words apart:
    // [leaf0_w0][leaf0_w1]...[leaf0_w7][leaf1_w0][leaf1_w1]...[leaf1_w7]...
    Layout LeavesLayout = make_layout(
        make_shape(Int<blake3::CHAINING_VALUE_SIZE_U32>{}, params.num_leaves),
        make_stride(Int<1>{}, Int<blake3::CHAINING_VALUE_SIZE_U32>{}));
    Tensor mLeaves = make_tensor(params.ptr_roots, LeavesLayout);
    Tensor sLeaves = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(reinterpret_cast<uint32_t*>(smem_buf)), LeavesLayout));
    // Each thread with a valid index copies its leaf to SMEM
    if (tid < params.num_leaves) {
      copy(mLeaves(_, tid), sLeaves(_, tid));
    }
    uint32_t rKey[blake3::CHAINING_VALUE_SIZE_U32];
    load_key(params.ptr_key, rKey);
    // Synchronize before starting the reduction
    __syncthreads();
    // And run the Merkle Tree reduction
    if (__popc(params.num_leaves) == 1) {
      merkle_tree_utils::compute_perfect_mt<true>(sLeaves, params.num_leaves,
                                                  rKey);
    } else {
      merkle_tree_utils::compute_blake_mt<true>(sLeaves, params.num_leaves, rKey);
    }
    // Copy the result back from smem -> gmem
    // Use the first 8 threads to write back the 8 uint32_t values in parallel
    if (tid < blake3::CHAINING_VALUE_SIZE_U32) {
      *((uint32_t*)params.ptr_roots + tid) = sLeaves(tid, 0);
      store_digest_word(params.ptr_out, tid, sLeaves(tid, 0));
    }
  }
};
}  // namespace pearl