#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <string>
#include <fstream>
#include <vector>

using namespace std;

void write_csv(std::string filename, std::string colname, std::vector<double> vals){
// taken from: https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
    std::ofstream myFile(filename);
    myFile << colname << "\n";
    for(int i = 0; i < vals.size(); ++i)
    {
        myFile << vals.at(i) << "\n";
    }
    myFile.close();
}

int main( int argc, char** argv )
{

  cv::Mat_<cv::Vec3b> source = cv::imread ( argv[1], cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> destination ( source.rows, source.cols/2 );

  // command line arguments for gaussian filter 
  int kernel_size_div2 = stof(argv[2]);
  double sigma = stof(argv[3]);

  //cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();

  const int iter = 1;
  int width = source.size().width;
  int height = source.size().height;

////////////////////////////////////////////////////////////
// Gaussian Blur Filter:
////////////////////////////////////////////////////////////

//int kernel_size_div2 = 11;
//double sigma = 1.0;
//cout << "kernel_size_div2: " << kernel_size_div2 << " s" << endl;
//cout << "sigma: " << sigma<< " s" << endl;

for (int it=0;it<iter;it++)
  {
  #pragma omp parallel for
    for (int i=0;i<source.rows;i++)
      for (int j=0;j<source.cols/2;j++)
        for (int c=0;c<3;c++)
        {
          double gaus_sum = 1;
          double dest_tmp = 0; 
          for (int ii=-1*kernel_size_div2; ii<=kernel_size_div2; ii++)
            for (int jj=-1*kernel_size_div2; jj<=kernel_size_div2; jj++)
            {
              double gaus_mult_term = 1.0 / 2.0 * M_PI * pow(sigma,2);
              double gaus_exp_term = -1.0 * ((pow(ii,2))+(pow(jj,2))) / (2.0*pow(sigma,2));
              double gaus = gaus_mult_term * exp(gaus_exp_term);
              gaus_sum += gaus;
              double thing = gaus*source(i+ii,j+jj)[c];
              dest_tmp += thing;
            }
          destination(i,j)[c] = dest_tmp/gaus_sum;
        }
  }


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Blur Image", destination );

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;
  
  cv::waitKey();

  std::vector<double> vec = { diff.count(), diff.count()/iter, iter/diff.count()};
  write_csv("times.csv", "anaglyphs and blur", vec);

  return 0;
}

