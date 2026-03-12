# Maintainer: rocm-polaris project
# Based on extra/rocblas PKGBUILD by Torsten Keßler and Christian Heusel
#
# Rebuilds rocBLAS with gfx803 (Polaris/Fiji) added back to the GPU target
# list. All other ROCm components use Arch's packages unmodified — their
# source code still supports gfx8xx, only rocBLAS dropped it from the
# CMake build configuration at ROCm 6.0.

pkgname=rocblas-gfx803
_upstream_pkgname=rocblas
pkgver=7.2.0
pkgrel=1
pkgdesc='rocBLAS with GCN 1.2 (gfx803) support restored'
arch=('x86_64')
url='https://github.com/slartibardfast/rocm-polaris'
license=('MIT')
depends=(
  'cblas'
  'gcc-libs'
  'glibc'
  'hip-runtime-amd'
  'openmp'
  'rocm-core'
  'roctracer'
)
makedepends=(
  'cmake'
  'gcc-fortran'
  'git'
  'msgpack-cxx'
  'perl-file-which'
  'python'
  'python-joblib'
  'python-msgpack'
  'python-pyaml'
  'python-tensile'
  'python-virtualenv'
  'python-wheel'
  'rocm-cmake'
  'rocm-toolchain'
  'rocm-smi-lib'
)
provides=("rocblas=$pkgver")
conflicts=('rocblas')
source=(
  "rocm-libraries-$pkgver.tar.gz::https://github.com/ROCm/rocm-libraries/archive/refs/tags/rocm-$pkgver.tar.gz"
  '0001-re-enable-gfx803-target.patch'
)
sha256sums=(
  '8ad5f4a11f1ed8a7b927f2e65f24083ca6ce902a42021a66a815190a91ccb654'
  'SKIP'
)
options=(!strip)
_dirname="rocm-libraries-rocm-$pkgver/projects/$_upstream_pkgname"
_tensile_dir="rocm-libraries-rocm-$pkgver/shared/tensile"

prepare() {
  cd "$_dirname"
  patch -p1 -i "$srcdir/0001-re-enable-gfx803-target.patch"
}

build() {
  # Compile source code for supported GPU archs in parallel
  export HIPCC_COMPILE_FLAGS_APPEND="-parallel-jobs=$(nproc)"
  export HIPCC_LINK_FLAGS_APPEND="-parallel-jobs=$(nproc)"

  # -fcf-protection is not supported by HIP, see
  # https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html#support-status-of-other-clang-options
  local cmake_args=(
    -Wno-dev
    -S "$_dirname"
    -B build
    -D CMAKE_BUILD_TYPE=RelWithDebInfo
    -D CMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/amdclang
    -D CMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/amdclang++
    -D CMAKE_TOOLCHAIN_FILE=toolchain-linux.cmake
    -D CMAKE_CXX_FLAGS="${CXXFLAGS} -fcf-protection=none"
    -D CMAKE_INSTALL_PREFIX=/opt/rocm
    -D CMAKE_PREFIX_PATH=/opt/rocm/llvm/lib/cmake/llvm
    -D amd_comgr_DIR=/opt/rocm/lib/cmake/amd_comgr
    -D BUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF
    -D HIP_PLATFORM=amd
    -D BLAS_LIBRARY=cblas
    -D BUILD_WITH_TENSILE=ON
    -D Tensile_LIBRARY_FORMAT=msgpack
    -D Tensile_TEST_LOCAL_PATH="$srcdir/$_tensile_dir"
    -D Tensile_COMPILER=hipcc
    -D BUILD_WITH_PIP=OFF
    -D BUILD_WITH_HIPBLASLT=OFF
  )
  cmake "${cmake_args[@]}"
  cmake --build build
}

package() {
  DESTDIR="$pkgdir" cmake --install build
  install -Dm644 "$_dirname/LICENSE.md" "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
