#include <test.hpp>
#include <vectorfield.hpp>
#include <kernel_dot.hpp>

namespace Test
{
    double testfunction()
    {
        vectorfield v1(10, Vector3{ 1.0, 1.0, 1.0 });
        vectorfield v2(10, Vector3{ -1.0, 1.0, 1.0 });

        scalar x = Kernel::dot(v1,v2);

        return x;
    } 
}