#ifndef USE_CUDA

#include <kernel_dot.hpp>

// C++ Version
namespace Kernel
{
    scalar dot(const vectorfield & v1, const vectorfield & v2)
    {
        scalar x=0;
        for (unsigned int i=0; i<v1.size(); ++i)
        {
            x += v1[i].dot(v2[i]);
        }
        return x;
    }
}

#endif