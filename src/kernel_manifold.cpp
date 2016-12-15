#ifndef USE_CUDA

#include <kernel_dot.hpp>
#include <kernel_manifold.hpp>

// C++ Version
namespace Kernel
{
	scalar norm(const vectorfield & v1)
	{
		scalar x = dot(v1, v1);
		return std::sqrt(x);
	}

	void normalize(vectorfield & v1)
	{
		scalar x = 1.0/norm(v1);
		for (unsigned int i = 0; i < v1.size(); ++i)
		{
			v1[i] *= x;
		}
	}

    void project_parallel(vectorfield & v1, const vectorfield & v2)
    {
        vectorfield v3 = v1;
        project_orthogonal(v3, v2);
		for (unsigned int i = 0; i < v1.size(); ++i)
		{
			v1[i] -= v3[i];
		}
    }

    void project_orthogonal(vectorfield & v1, const vectorfield & v2)
    {
        scalar x=dot(v1, v2);
        for (unsigned int i=0; i<v1.size(); ++i)
        {
            v1[i] -= x*v2[i];
        }
    }

    void invert_parallel(vectorfield & v1, const vectorfield & v2)
    {
        scalar x=dot(v1, v2);
        for (unsigned int i=0; i<v1.size(); ++i)
        {
            v1[i] -= 2*x*v2[i];
        }
    }
    
    void invert_orthogonal(vectorfield & v1, const vectorfield & v2)
    {
        vectorfield v3 = v1;
        project_orthogonal(v3, v2);
		for (unsigned int i = 0; i < v1.size(); ++i)
		{
			v1[i] -= 2 * v3[i];
		}
    }
}

#endif