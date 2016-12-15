# vectorfield
Kernels acting on scalar- and vectorfields

## Project Health

| Service | System | Compiler | Status |
| ------- | ------ | -------- | -----: |
|  [Travis-CI](https://travis-ci.org/GPMueller/vectorfield) | Ubuntu 14.04 | GCC 6 | [![Build Status](https://travis-ci.org/GPMueller/vectorfield.svg?branch=master)](https://travis-ci.org/GPMueller/vectorfield) |
|  [Travis-CI](https://travis-ci.org/GPMueller/vectorfield) | macOS | Clang ? | upcoming |

The vectorfields are setup as `std::vector<Eigen::Vector3d>` with optional CUDA support.
If CUDA is used, an allocator using cudaMallocManaged is employed so that the user
does not have to worry about copying vectorfields between host and device, except for
performance.
