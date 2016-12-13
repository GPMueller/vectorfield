#include <vector>
#include <Eigen/Core>

typedef double scalar;
typedef Eigen::Matrix<scalar, 3, 1> Vector3;

#ifdef USE_CUDA
    #include "managed_allocator.hpp"
    typedef std::vector<Vector3, managed_allocator<Vector3>> vectorfield;
    typedef std::vector<scalar, managed_allocator<scalar>> scalarfield;
#else
    typedef std::vector<Vector3> vectorfield;
    typedef std::vector<scalar> scalarfield;
#endif