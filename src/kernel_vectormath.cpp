#ifndef USE_CUDA

#include <kernel_vectormath.hpp>
#include <Eigen/Dense>

// C++ Version
namespace Kernel
{
    // sets vf := v
    // vf is a vectorfield
    // v is a vector
    void fill(vectorfield & vf, const Vector3 & v)
    {
        for (unsigned int i=0; i<vf.size(); ++i)
        {
            vf[i] = v;
        }
    }

	// computes the inner product of two vectorfields v1 and v2
	scalar dot(const vectorfield & v1, const vectorfield & v2)
	{
		scalar x = 0;
		for (unsigned int i = 0; i<v1.size(); ++i)
		{
			x += v1[i].dot(v2[i]);
		}
		return x;
	}

    // computes the inner products of vectors in v1 and v2
    // v1 and v2 are vectorfields
    void dot(const vectorfield & v1, const vectorfield & v2, scalarfield & out)
    {
        for (unsigned int i=0; i<v1.size(); ++i)
        {
			out[i] = v1[i].dot(v2[i]);
        }
    }

    // computes the vector (cross) products of vectors in v1 and v2
    // v1 and v2 are vector fields
    void cross(const vectorfield & v1, const vectorfield & v2, vectorfield & out)
    {
        for (unsigned int i=0; i<v1.size(); ++i)
        {
            out[i] = v1[i].cross(v2[i]);
        }
    }


    // out[i] += c*a
    void add_c_a(const scalar & c, const Vector3 & a, vectorfield & out)
    {
        for(unsigned int idx = 0; idx < out.size(); ++idx)
        {
            out[idx] += c*a;
        }
    }

    // out[i] += c*a[i]
    void add_c_a(const scalar & c, const vectorfield & a, vectorfield & out)
    {
        for(unsigned int idx = 0; idx < out.size(); ++idx)
        {
            out[idx] += c*a[idx];
        }
    }


    // out[i] += c * a*b[i]
    void add_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out)
    {
        for(unsigned int idx = 0; idx < out.size(); ++idx)
        {
            out[idx] += c*a.dot(b[idx]);
        }
    }

    // out[i] += c * a[i]*b[i]
    void add_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out)
    {
        for(unsigned int idx = 0; idx < out.size(); ++idx)
        {
            out[idx] += c*a[idx].dot(b[idx]);
        }
    }


    // out[i] += c * a x b[i]
    void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
    {
        for(unsigned int idx = 0; idx < out.size(); ++idx)
        {
            out[idx] += c*a.cross(b[idx]);
        }
    }

    // out[i] += c * a[i] x b[i]
    void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
    {
        for(unsigned int idx = 0; idx < out.size(); ++idx)
        {
            out[idx] += c*a[idx].cross(b[idx]);
        }
    }

}

#endif