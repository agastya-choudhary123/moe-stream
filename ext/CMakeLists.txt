cmake_minimum_required(VERSION 3.25)
project(mlx_zerocopy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(
  Python 3.12
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# nanobind ships its cmake config inside the installed package
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NB_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# MLXConfig.cmake lives in the installed mlx package
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import mlx, os; print(os.path.join(mlx.__path__[0], 'share', 'cmake', 'MLX'))"
  OUTPUT_VARIABLE MLX_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${MLX_CMAKE_DIR}")
find_package(MLX CONFIG REQUIRED)

message(STATUS "MLX includes : ${MLX_INCLUDE_DIRS}")
message(STATUS "MLX library  : ${MLX_LIBRARY}")

# NB_DOMAIN must match the domain MLX's own bindings use, otherwise nanobind
# treats mlx::core::array as a different type and cannot cast it across modules.
nanobind_add_module(mlx_zerocopy_ext NB_DOMAIN mlx src/zerocopy.cpp)

target_include_directories(mlx_zerocopy_ext PRIVATE ${MLX_INCLUDE_DIRS})
target_link_libraries(mlx_zerocopy_ext PRIVATE ${MLX_LIBRARY})
target_compile_options(mlx_zerocopy_ext PRIVATE ${MLX_CXX_FLAGS})
