#include <catch.hpp>
#include <vectorfield.hpp>
#include <kernel_vectormath.hpp>
#include <kernel_manifold.hpp>


TEST_CASE( "Vectormath operations", "[vectormath]" )
{
    int N = 10000;
    vectorfield v1(N, Vector3{ 1.0, 1.0, 1.0 });
    vectorfield v2(N, Vector3{ -1.0, 1.0, 1.0 });
	
	SECTION("Dot Product")
	{
		scalarfield dots(N, N);
		Kernel::dot(v1,v2, dots);
		REQUIRE( dots[0] == Approx(1) );
		REQUIRE( Kernel::dot(v1,v2) == Approx(N) );
	}

	// SECTION("Add c*a")
	// {
	// }
	// SECTION("Add c*v1.dot(v2)")
	// {
	// }
	// SECTION("Add c*v1.cross(v2)")
	// {
	// }
}

TEST_CASE( "Manifold operations", "[manifold]" )
{
	int N = 10000;
	vectorfield v1(N, Vector3{ 0.0, 0.0, 1.0 });
	vectorfield v2(N, Vector3{ 1.0, 1.0, 1.0 });

	REQUIRE( Kernel::dot(v1,v2) == Approx(N) );
	Kernel::normalize(v1);
	Kernel::normalize(v2);

	SECTION("Normalisation")
	{
		REQUIRE( Kernel::dot(v1, v1) == Approx(1) );
		REQUIRE( Kernel::norm(v1) == Approx(1) );
		REQUIRE( Kernel::norm(v2) == Approx(1) );
	}

	SECTION("Projection: parallel")
	{
		Kernel::project_parallel(v1,v2);
		REQUIRE( Kernel::dot(v1,v2) == Approx(Kernel::norm(v1)*Kernel::norm(v2)) );
	}

	SECTION("Projection: orthogonal")
	{
		Kernel::project_orthogonal(v1,v2);
		REQUIRE( Kernel::dot(v1,v2) == Approx(0) );
	}

	SECTION("Invert: parallel")
	{
		scalar proj_prev = Kernel::dot(v1, v2);
		Kernel::invert_parallel(v1,v2);
		REQUIRE( Kernel::dot(v1,v2) == Approx(-proj_prev) );
	}

	SECTION("Invert: orthogonal")
	{
		vectorfield v3 = v1;
		Kernel::project_orthogonal(v3, v2);
		scalar proj_prev = Kernel::dot(v1, v3);
		Kernel::invert_orthogonal(v1,v2);
		REQUIRE( Kernel::dot(v1, v3) == Approx(-proj_prev) );
	}
}