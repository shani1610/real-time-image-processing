#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernel_size_div2, double sigma )
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  // gaussian filter:
  if (dst_x < cols && dst_y < rows)
    {
      double sum_x = 0.0;
      double sum_y = 0.0;
      double sum_z = 0.0;
      double factor = 0.0;
      //double sigma = 1.0;
      //int kernel_size_div2 = 3;
      for (int i = -1*kernel_size_div2; i<kernel_size_div2+1; i++)
        for (int j = -1*kernel_size_div2; j<kernel_size_div2+1; j++) 
        {
          uchar3 val = src(dst_y+j, dst_x+i);
          double gaus_mult_term = 1.0 / 2.0 * M_PI * pow(sigma,2);
          double gaus_exp_term = -1.0 * ((pow(i,2))+(pow(j,2))) / (2.0*pow(sigma,2));
          double gaus = gaus_mult_term * exp(gaus_exp_term);
          factor += gaus;
          sum_x += gaus*val.x;
          sum_y += gaus*val.y;
          sum_z += gaus*val.z;
        }
      dst(dst_y, dst_x).x = sum_x/factor; //b
      dst(dst_y, dst_x).y = sum_y/factor; //g
      dst(dst_y, dst_x).z = sum_z/factor; //r
    }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size_div2, double sigma  )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, kernel_size_div2, sigma);


}

