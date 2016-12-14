#include <catch.hpp>
#include <vectorfield.hpp>
#include <kernel_dot.hpp>


TEST_CASE( "N-dimensional dot product", "[dot]" )
{
    int N = 10000;
    vectorfield v1(N, Vector3{ 1.0, 1.0, 1.0 });
    vectorfield v2(N, Vector3{ -1.0, 1.0, 1.0 });
    REQUIRE( Kernel::dot(v1,v2) == N );
    REQUIRE( Kernel::dot(v1,v2)-1 == N-1 );
}