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
  int cov_kernel_size_div2 = atof(argv[2]);
  double factor_ratio = atof(argv[3]);
  //double sigma = stof(argv[3]);

  //cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();

  const int iter = 1;
  int width = source.size().width;
  int height = source.size().height;

////////////////////////////////////////////////////////////
// Denoising Filter:
////////////////////////////////////////////////////////////

double cov_mat[3][3];
double mean[3];
double minor1 = 0.0;
double minor2 = 0.0;
double minor3 = 0.0;
double det = 0.0;
int kernel_size_div2 = 1;
double sigma = 1.0;

for (int it=0;it<iter;it++)
  {
  #pragma omp parallel for

    for (int i=0;i<source.rows;i++)
      for (int j=0;j<source.cols/2;j++)
      {

        // initialize the mean array
        for (int c0=0;c0<3;c0++)
        {
          mean[c0] = 0.0;
        }

        // calculate the mean for each channel
        for (int c=0;c<3;c++)
        {
            for (int ii=-1*cov_kernel_size_div2; ii<=cov_kernel_size_div2; ii++)
                for (int jj=-1*cov_kernel_size_div2; jj<=cov_kernel_size_div2; jj++)
                {
                    mean[c] += source(i+ii,j+jj)[c];
                }
            mean[c] = mean[c]/(2*cov_kernel_size_div2+1);
        }

        // initialize the covariance mat
        for (int ii1=0; ii1<3; ii1++)
          for (int jj1=0; jj1<3; jj1++)
          {
            cov_mat[ii1][jj1] = 0.0;
          }

        // calculate the covariance matrix
        for (int ii2=-1*cov_kernel_size_div2; ii2<=cov_kernel_size_div2; ii2++)
          for (int jj2=-1*cov_kernel_size_div2; jj2<=cov_kernel_size_div2; jj2++)
          {
            cov_mat[0][0] += (source(i+ii2,j+jj2)[2]-mean[2])*(source(i+ii2,j+jj2)[2]-mean[2]); // rr
            cov_mat[0][1] += (source(i+ii2,j+jj2)[2]-mean[2])*(source(i+ii2,j+jj2)[1]-mean[1]); // rg
            cov_mat[0][2] += (source(i+ii2,j+jj2)[2]-mean[2])*(source(i+ii2,j+jj2)[0]-mean[0]); // rb

            cov_mat[1][0] += (source(i+ii2,j+jj2)[1]-mean[1])*(source(i+ii2,j+jj2)[2]-mean[2]); // gr
            cov_mat[1][1] += (source(i+ii2,j+jj2)[1]-mean[1])*(source(i+ii2,j+jj2)[1]-mean[1]); // gg
            cov_mat[1][2] += (source(i+ii2,j+jj2)[1]-mean[1])*(source(i+ii2,j+jj2)[0]-mean[0]); // gb

            cov_mat[2][0] += (source(i+ii2,j+jj2)[0]-mean[0])*(source(i+ii2,j+jj2)[2]-mean[2]); // br
            cov_mat[2][1] += (source(i+ii2,j+jj2)[0]-mean[0])*(source(i+ii2,j+jj2)[1]-mean[1]); // bg
            cov_mat[2][2] += (source(i+ii2,j+jj2)[0]-mean[0])*(source(i+ii2,j+jj2)[0]-mean[0]); // bb
          }
        
        // scaling the covariance matrix: 
        for (int ii3=0; ii3<3; ii3++)
          for (int jj3=0; jj3<3; jj3++)
            {
              cov_mat[ii3][jj3] /= pow(2,cov_kernel_size_div2); 
            }
        
        // calculating the determintant
        minor1 = cov_mat[1][1]*cov_mat[2][2]-cov_mat[1][2]*cov_mat[2][1];
        minor2 = cov_mat[1][0]*cov_mat[2][2]-cov_mat[1][2]*cov_mat[2][0];
        minor3 = cov_mat[1][0]*cov_mat[2][1]-cov_mat[1][1]*cov_mat[2][0];
        det = cov_mat[0][0]*minor1+cov_mat[0][1]*minor2+cov_mat[0][2]*minor3;
        //det = cv::determinant(cov_mat);

        // // finding the gaussian size 
        int potential_kernel = (det*3)/(factor_ratio*1000000000000);
        if (potential_kernel > 21 ){
            kernel_size_div2 = 21;
        }
        else {
            kernel_size_div2 = potential_kernel;
        }
        kernel_size_div2 = 3;

        // applying gaussian
        for (int c2=0;c2<3;c2++)
        {
          double gaus_sum = 1;
          double dest_tmp = 0; 
          for (int ii4=-1*kernel_size_div2; ii4<=kernel_size_div2; ii4++)
            for (int jj4=-1*kernel_size_div2; jj4<=kernel_size_div2; jj4++)
            {
              double gaus_mult_term = 1.0 / 2.0 * M_PI * pow(sigma,2);
              double gaus_exp_term = -1.0 * ((pow(ii4,2))+(pow(jj4,2))) / (2.0*pow(sigma,2));
              double gaus = gaus_mult_term * exp(gaus_exp_term);
              gaus_sum += gaus;
              double thing = gaus*source(i+ii4,j+jj4)[c2];
              dest_tmp += thing;
            }
          destination(i,j)[c2] = dest_tmp/gaus_sum;
        }
      }
  }


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Denoised Image", destination );

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;
  
  cv::waitKey();

  std::vector<double> vec = { diff.count(), diff.count()/iter, iter/diff.count()};
  write_csv("times.csv", "anaglyphs and blur", vec);

  return 0;
}

