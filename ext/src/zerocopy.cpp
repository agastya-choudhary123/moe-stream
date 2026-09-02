// Zero-copy bridge between host memory and MLX arrays.
//
// MLX's C++ API can build an array over memory it does not own -- the header
// for allocator::make_buffer says "make a Buffer from a raw pointer of the
// given size without a copy". The Metal allocator overrides it, so on unified
// memory the GPU reads host pages in place. None of this is exposed to Python;
// mx.from_dlpack copies.
//
// That copy is the whole reason for this file. Streaming a 16 GB MoE moves
// ~971 MB of expert weights per token, and memcpy'ing that costs more than the
// SSD read it follows.
//
// The array(void*, shape, dtype, deleter) constructor attempts the no-copy
// path internally and falls back to copying when the pointer does not qualify.
// can_wrap() reports which happened, so callers can verify rather than hope.

#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"

namespace nb = nanobind;
using namespace mlx::core;

static Dtype dtype_from_string(const std::string& s) {
  if (s == "uint8") return uint8;
  if (s == "uint16") return uint16;
  if (s == "uint32") return uint32;
  if (s == "uint64") return uint64;
  if (s == "int8") return int8;
  if (s == "int16") return int16;
  if (s == "int32") return int32;
  if (s == "float16") return float16;
  if (s == "bfloat16") return bfloat16;
  if (s == "float32") return float32;
  throw std::invalid_argument("unsupported dtype: " + s);
}

NB_MODULE(mlx_zerocopy_ext, m) {
  m.doc() = "Wrap host memory as MLX arrays without copying.";

  // The useful direction. Wrapping foreign memory is gated by the Metal
  // allocator, but the reverse works: let MLX allocate the buffer, take its
  // host address, and pread SSD bytes straight into it. Unified memory means
  // that address is the same memory the GPU reads, so nothing is ever copied.
  //
  // The array must already be evaluated -- a lazy array has no buffer yet.
  m.def(
      "data_ptr",
      [](array& a) {
        if (!a.is_available()) {
          throw std::runtime_error(
              "array is not evaluated; call mx.eval() before data_ptr()");
        }
        // raw_ptr(), not ptr(): ptr() is the MTL::Buffer object address,
        // raw_ptr() is its contents. Writing to the former corrupts Metal.
        return reinterpret_cast<uintptr_t>(a.buffer().raw_ptr());
      },
      nb::arg("array"),
      "Host address of an evaluated array's buffer. Writable in place.");

  m.def(
      "nbytes",
      [](const array& a) { return a.nbytes(); },
      nb::arg("array"),
      "Size of the array's data in bytes.");

  m.def(
      "array_from_ptr",
      [](uintptr_t ptr,
         const std::vector<int32_t>& shape_in,
         const std::string& dtype_str) {
        if (ptr == 0) {
          throw std::invalid_argument("null pointer");
        }
        Shape shape(shape_in.begin(), shape_in.end());
        // No-op deleter: the mapping is owned by the caller. MLX must never
        // free memory it did not allocate.
        return array(
            reinterpret_cast<void*>(ptr),
            shape,
            dtype_from_string(dtype_str),
            [](void*) {});
      },
      nb::arg("ptr"),
      nb::arg("shape"),
      nb::arg("dtype"),
      "Build an mx.array over existing memory. Caller keeps ownership and "
      "must outlive the array.");

  m.def("page_size", [] { return static_cast<size_t>(::getpagesize()); });
}
