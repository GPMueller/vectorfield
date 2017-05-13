# vectorfield
Kernels acting on scalar- and vectorfields

## Project Health

| Service | System | Compiler | Status |
| ------- | ------ | -------- | -----: |
|  [Travis-CI](https://travis-ci.org/GPMueller/vectorfield) | Ubuntu 14.04 <br/> macOS | GCC 6 <br/> Clang | [![Build Status](https://travis-ci.org/GPMueller/vectorfield.svg?branch=master)](https://travis-ci.org/GPMueller/vectorfield) |
|  [AppVeyor](https://ci.appveyor.com/project/GPMueller/vectorfield) | Windows | MSVC12 <br/> MSVC14 <br/> MSVC14.1 | [![Build status](https://ci.appveyor.com/api/projects/status/dtiflux1wmhm6sd2?svg=true)](https://ci.appveyor.com/project/GPMueller/vectorfield) |

The vectorfields are setup as `std::vector<Eigen::Vector3d>` with optional CUDA support.
If CUDA is used, an allocator using cudaMallocManaged is employed so that the user
does not have to worry about copying vectorfields between host and device, except for
performance.
