# Vendored header-only SPLBATCH; version pinned in VENDORED.md.
add_library(slict_splbatch INTERFACE)
add_library(splbatch::splbatch ALIAS slict_splbatch)
target_compile_features(slict_splbatch INTERFACE cxx_std_17)
target_include_directories(slict_splbatch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(slict_splbatch INTERFACE Eigen3::Eigen Ceres::ceres)
