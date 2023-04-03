#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <vector>
#include <bits/stdc++.h>
using namespace std;
#define N 3
#include "helper_math.h"

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, const int cov_kernel_size_div2, int factor_ratio )
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  const int max_kernel_size_div2 = 13;
if (dst_x < cols && dst_y < rows)
  {
        double mean_r = 0.0;
        double mean_g = 0.0;
        double mean_b = 0.0;
        double covariance_mat[3][3];
        double minor1 = 0.0;
        double minor2 = 0.0;
        double minor3 = 0.0;
        double det = 0.0;

        // initializing the covariance matrix to zeros
        for (int jj = 0; jj<3; jj++)
        {
          for (int ii = 0; ii<3; ii++) 
          { 
            covariance_mat[jj][ii] = 0.0;
          }
        }

        // calculating the mean of each channel:
        for (int j = -1*cov_kernel_size_div2; j<cov_kernel_size_div2+1; j++)
        {
          for (int i = -1*cov_kernel_size_div2; i<cov_kernel_size_div2+1; i++) 
          { 
            uchar3 val = src(dst_y+j, dst_x+i);
            mean_r += val.z; // z corresponds to red
            mean_g += val.y;
            mean_b += val.x;
          }
          mean_r = mean_r/(2*cov_kernel_size_div2+1);
          mean_g = mean_g/(2*cov_kernel_size_div2+1);
          mean_b = mean_b/(2*cov_kernel_size_div2+1);
        }

        // calculating the covariance matrix :
        for (int j = -1*cov_kernel_size_div2; j<cov_kernel_size_div2+1; j++)
        {
          for (int i = -1*cov_kernel_size_div2; i<cov_kernel_size_div2+1; i++) 
          { 
            uchar3 val = src(dst_y+j, dst_x+i);
            // First line of covariance matrix
            covariance_mat[0][0] += (val.z - mean_r)*(val.z - mean_r); // RR 
            covariance_mat[0][1] += (val.z - mean_r)*(val.y - mean_g); // RG
            covariance_mat[0][2] += (val.z - mean_r)*(val.x - mean_b); // RB
            // Second line of covariance matrix
            covariance_mat[1][0] += (val.y - mean_g)*(val.z - mean_r); // GR 
            covariance_mat[1][1] += (val.y - mean_g)*(val.y - mean_g); // GG 
            covariance_mat[1][2] += (val.y - mean_g)*(val.x - mean_b); // GB
            // Third line of covariance matrix
            covariance_mat[2][0] += (val.x - mean_b)*(val.z - mean_r); // BR
            covariance_mat[2][1] += (val.x - mean_b)*(val.y - mean_g); // BG 
            covariance_mat[2][2] += (val.x - mean_b)*(val.x - mean_b); // BB 

          }
        }

        // scaling the covariance matrix
        for (int jj = 0; jj<3; jj++)
        {
          for (int ii = 0; ii<3; ii++) 
          { 
            covariance_mat[jj][ii] *= (1/(2*cov_kernel_size_div2+1));
          }
        }

        // calculating the covariance matrix determinant (for mat size 3)
        minor1 = covariance_mat[1][1]*covariance_mat[2][2]-covariance_mat[1][2]*covariance_mat[2][1];
        minor2 = covariance_mat[1][0]*covariance_mat[2][2]-covariance_mat[1][2]*covariance_mat[2][0];
        minor3 = covariance_mat[1][0]*covariance_mat[2][1]-covariance_mat[1][1]*covariance_mat[2][0];
        det = covariance_mat[0][0]*minor1+covariance_mat[0][1]*minor2+covariance_mat[0][2]*minor3;
        double sigma = 1.0;
        int kernel_size_div2 = (det*3)/factor_ratio;

        // applying the gaussian filter
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        double factor = 0.0;

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

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const int cov_kernel_size_div2, int factor_ratio  )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, cov_kernel_size_div2, factor_ratio);


}

