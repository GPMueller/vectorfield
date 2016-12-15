#pragma once
#ifndef KERNEL_VECTORMATH_H
#define KERNEL_VECTORMATH_H

#include <vectorfield.hpp>

namespace Kernel
{
    
    // sets vf := v
    // vf is a vectorfield
    // v is a vector
    void fill(vectorfield & vf, const Vector3 & v);

	// computes the inner product of two vectorfields v1 and v2
	scalar dot(const vectorfield & v1, const vectorfield & v2);

    // computes the inner products of vectors in v1 and v2
    // v1 and v2 are vectorfields
    void dot(const vectorfield & v1, const vectorfield & v2, scalarfield & out);
    
    // computes the vector (cross) products of vectors in v1 and v2
    // v1 and v2 are vector fields
    void cross(const vectorfield & v1, const vectorfield & v2, vectorfield & out);
    
    // out[i] += c*a
	void add_c_a(const scalar & c, const Vector3 & a, vectorfield & out);

    // out[i] += c*a[i]
	void add_c_a(const scalar & c, const vectorfield & a, vectorfield & out);

    // out[i] += c * a*b[i]
    void add_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out);

    // out[i] += c * a[i]*b[i]
    void add_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out);

    // out[i] += c * a x b[i]
    void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out);

    // out[i] += c * a[i] x b[i]
    void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out);
    
}

#endif