#pragma once
#ifndef KERNEL_H
#define KERNEL_H

#include <vectorfield.hpp>

namespace Kernel
{
    scalar dot(const vectorfield & v1, const vectorfield & v2);
}

#endif