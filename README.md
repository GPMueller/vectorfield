# vectorfield
Kernels acting on vectorfields

The vectorfields are setup as std::vector<Eigen::Vector3d> with optional CUDA support.
If CUDA is used, an allocator using cudaMallocManaged is employed so that the user
does not have to worry about copying vectorfields between host and device, except for
performance.
