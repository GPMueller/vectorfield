#ifndef MANAGED_ALLOCATOR_H
#define MANAGED_ALLOCATOR_H

#ifdef USE_CUDA

#include <stdio.h>

static void HandleError( cudaError_t err, const char *file, int line )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
  {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


template<class T>
class managed_allocator
{
  public:
    using value_type = T;
  
    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

      HANDLE_ERROR( cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal) );

      return result;
    }
  
    void deallocate(value_type* ptr, size_t)
    {
      HANDLE_ERROR( cudaFree(ptr) );
    }
};

#endif
#endif