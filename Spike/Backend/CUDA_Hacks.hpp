#ifndef CUDA_HACKS_H
#define CUDA_HACKS_H

struct uint3
{
  unsigned int x, y, z;
};
typedef struct uint3 uint3;

struct dim3
{
  unsigned int x, y, z;
#if defined(__cplusplus)
  dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
  dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif /* __cplusplus */
};
typedef struct dim3 dim3;

#endif
