#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <iostream>

#include "helper_math.h"

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int method_num )
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x < cols && dst_y < rows)
    {
      double sum_b = 0.0;
      double sum_g = 0.0;
      double sum_r = 0.0;
      uchar3 val_r = src(dst_y, dst_x+cols);
      uchar3 val_l = src(dst_y, dst_x);
      if (method_num==1) // True
      {
          sum_b = 0.299*val_r.z+0.587*val_r.y+0.114*val_r.x;
          sum_g = 0.0;
          sum_r = 0.299*val_l.z+0.587*val_l.y+0.114*val_l.x;
      }
      else if (method_num==2) // Gray
      {
          sum_b = 0.299*val_r.z+0.587*val_r.y+0.114*val_r.x;
          sum_g = 0.299*val_r.z+0.587*val_r.y+0.114*val_r.x;
          sum_r = 0.299*val_l.z+0.587*val_l.y+0.114*val_l.x;
      }
      else if (method_num==3) // Color
      {
          sum_b = 0.0*val_r.z+0.0*val_r.y+1.0*val_r.x;
          sum_g = 0.0*val_r.z+1.0*val_r.y+0.0*val_r.x;
          sum_r = 1.0*val_l.z+0.0*val_l.y+0.0*val_l.x;
      }
      else if (method_num==4) // Half-Color
      {
          sum_b = 0.0*val_r.z+0.0*val_r.y+1.0*val_r.x;
          sum_g = 0.0*val_r.z+1.0*val_r.y+0.0*val_r.x;
          sum_r = 0.299*val_l.z+0.587*val_l.y+0.114*val_l.x;
      }
      else if (method_num==5) // Optimized
      {
          sum_b = 0.0*val_r.z+0.0*val_r.y+1.0*val_r.x;
          sum_g = 0.0*val_r.z+1.0*val_r.y+0.0*val_r.x;
          sum_r = 0.0*val_l.z+0.7*val_l.y+0.3*val_l.x;
      }
      dst(dst_y, dst_x).x = sum_b; //b
      dst(dst_y, dst_x).y = sum_g; //g
      dst(dst_y, dst_x).z = sum_r; //r
    }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int method_num )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, method_num);
}

