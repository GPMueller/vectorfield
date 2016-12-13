#ifdef USE_CUDA

#include <kernel_dot.hpp>

#include <iostream>
#include <stdio.h>

// CUDA Version
namespace Kernel
{
    __global__ void cu_dot(const Vector3 *v1, const Vector3 *v2, double *out, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
        {
            out[idx] = v1[idx].dot(v2[idx]);
        }
        return;
    }

    // The wrapper for the calling of the actual kernel
    scalar dot(const vectorfield & v1, const vectorfield & v2)
    {        
        int n = v1.size();
        scalarfield ret(n);

        // Dot product
        cu_dot<<<(n+1023)/1024, 1024>>>(v1.data(), v2.data(), ret.data(), n);
        cudaDeviceSynchronize();

        // Reduction of the array
        for (int i=1; i<n; ++i)
        {
            ret[0] += ret[i];
        }

        // Return
        return ret[0];
    }
}

#endif