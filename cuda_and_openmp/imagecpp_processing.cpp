#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst, int kernel_size_div2, double sigma);

int main( int argc, char** argv )
{
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  cv::Mat h_result(h_img.rows, h_img.cols/2, CV_8UC3);  
  
  const int kernel_size_div2 = stof(argv[2]);
  double sigma = stof(argv[3]);

  cv::cuda::GpuMat d_img, d_result;

  d_img.upload(h_img);
  d_result.upload(h_result);
  int width= d_img.cols;
  int height = d_img.rows;

  //cv::imshow("Original Image", h_img);
  
  auto begin = chrono::high_resolution_clock::now();
  //const int iter = 100000; // too long 
  const int iter = 100;

  for (int i=0;i<iter;i++)
    {
      startCUDA ( d_img, d_result, kernel_size_div2, sigma);
    }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  d_result.download(h_result);
  cv::imshow("Processed Image", h_result);

  cout << "Time: "<< diff.count() << endl;
  cout << "Time/frame: " << diff.count()/iter << endl;
  cout << "IPS: " << iter/diff.count() << endl;
  
  cv::waitKey();
  
  return 0;
}
