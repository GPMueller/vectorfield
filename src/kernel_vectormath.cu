#ifdef USE_CUDA

#include <kernel_vectormath.hpp>

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
    }

    // The wrapper for the calling of the actual kernel
    void dot(const vectorfield & v1, const vectorfield & v2, scalarfield & s)
    {        
        int n = v1.size();

        // Dot product
        cu_dot<<<(n+1023)/1024, 1024>>>(v1.data(), v2.data(), s.data(), n);
        cudaDeviceSynchronize();
    }

    __global__ void cu_cross(const Vector3 *v1, const Vector3 *v2, Vector3 *out, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
        {
            out[idx] = v1[idx].cross(v2[idx]);
        }
    }

    // The wrapper for the calling of the actual kernel
    void cross(const vectorfield & v1, const vectorfield & v2, vectorfield & s)
    {        
        int n = v1.size();

        // Dot product
        cu_cross<<<(n+1023)/1024, 1024>>>(v1.data(), v2.data(), s.data(), n);
        cudaDeviceSynchronize();
    }
}

#endif