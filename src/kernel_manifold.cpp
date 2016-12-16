#ifndef USE_CUDA

#include <kernel_vectormath.hpp>
#include <kernel_manifold.hpp>

// C++ Version
namespace Kernel
{
	scalar norm(const vectorfield & vf)
	{
		scalar x = dot(vf, vf);
		return std::sqrt(x);
	}

	void normalize(vectorfield & vf)
	{
		scalar x = 1.0/norm(vf);
		for (unsigned int i = 0; i < vf.size(); ++i)
		{
			vf[i] *= x;
		}
	}

    void project_parallel(vectorfield & vf1, const vectorfield & vf2)
    {
        vectorfield vf3 = vf1;
        project_orthogonal(vf3, vf2);
        // TODO: replace the loop with Vectormath Kernel
		for (unsigned int i = 0; i < vf1.size(); ++i)
		{
			vf1[i] -= vf3[i];
		}
    }

    void project_orthogonal(vectorfield & vf1, const vectorfield & vf2)
    {
        scalar x=dot(vf1, vf2);
        // TODO: replace the loop with Vectormath Kernel
        for (unsigned int i=0; i<vf1.size(); ++i)
        {
            vf1[i] -= x*vf2[i];
        }
    }

    void invert_parallel(vectorfield & vf1, const vectorfield & vf2)
    {
        scalar x=dot(vf1, vf2);
        // TODO: replace the loop with Vectormath Kernel
        for (unsigned int i=0; i<vf1.size(); ++i)
        {
            vf1[i] -= 2*x*vf2[i];
        }
    }
    
    void invert_orthogonal(vectorfield & vf1, const vectorfield & vf2)
    {
        vectorfield vf3 = vf1;
        project_orthogonal(vf3, vf2);
        // TODO: replace the loop with Vectormath Kernel
		for (unsigned int i = 0; i < vf1.size(); ++i)
		{
			vf1[i] -= 2 * vf3[i];
		}
    }
}

#endif