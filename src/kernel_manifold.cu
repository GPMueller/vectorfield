#ifdef USE_CUDA

#include <kernel_vectormath.hpp>

#include <iostream>
#include <stdio.h>

// CUDA Version
namespace Kernel
{
    __global__ void cu_project_orthogonal(const Vector3 *v1, const Vector3 *v2, scalar proj, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < N)
        {
            v1[idx] -= proj*v2[idx];
        }
    }

    // The wrapper for the calling of the actual kernel
    void project_orthogonal(const vectorfield & v1, const vectorfield & v2)
    {        
        int n = v1.size();
        scalarfield ret(n);

        // Get projection
        scalar proj=dot(v1, v2);
        // Project v1
        cu_project_orthogonal<<<(n+1023)/1024, 1024>>>(v1.data(), v2.data(), proj, n);
        cudaDeviceSynchronize();
    }
}

#endif