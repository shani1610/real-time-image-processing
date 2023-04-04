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
  int anaglyph_method = stof(argv[2]);

  // cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();

  const int iter = 1;
  int width = source.size().width;
  int height = source.size().height;

////////////////////////////////////////////////////////////
// Anaglyph Methods:
////////////////////////////////////////////////////////////
  for (int it=0;it<iter;it++)
    {
	 #pragma omp parallel for
      for (int i=0;i<source.rows;i++)
	      for (int j=0;j<source.cols/2;j++)
        {
            double r1 = source(i,j)[2];
            double g1 = source(i,j)[1];
            double b1 = source(i,j)[0];
            double r2 = source(i,j+source.cols/2)[2];
            double g2 = source(i,j+source.cols/2)[1];
            double b2 = source(i,j+source.cols/2)[0];
            if (anaglyph_method == 0){ //True
              destination(i,j)[0] = 0.249*r2+0.587*g2+0.114*b2; // b
              destination(i,j)[1] = 0; // g
              destination(i,j)[2] = 0.249*r1+0.587*g1+0.114*b1; // r
            }
            else if (anaglyph_method == 1){ // Gray 
              destination(i,j)[0] = 0.249*r2+0.587*g2+0.114*b2; // b
              destination(i,j)[1] = 0.249*r2+0.587*g2+0.114*b2; // g
              destination(i,j)[2] = 0.249*r1+0.587*g1+0.114*b1; // r
            }
            else if (anaglyph_method == 2){ // Color
              destination(i,j)[0] = 1.0*b2; // b
              destination(i,j)[1] = 1.0*g2; // g
              destination(i,j)[2] = 1.0*r1; // r
            }
            else if (anaglyph_method == 3){ // Half-Color
              destination(i,j)[0] = 1.0*b2; // b
              destination(i,j)[1] = 1.0*g2; // g
              destination(i,j)[2] = 0.249*r1+0.587*g1+0.114*b1; // r
            }
            else if (anaglyph_method == 4){ // Optimized 
              destination(i,j)[0] = 1.0*b2; // b
              destination(i,j)[1] = 1.0*g2; // g
              destination(i,j)[2] = 0.7*g1+0.3*b1; // r
            }
        }
    }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Anaglyph Image", destination );

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;
  
  cv::waitKey();

  std::vector<double> vec = { diff.count(), diff.count()/iter, iter/diff.count()};
  write_csv("times.csv", "anaglyphs and blur", vec);

  return 0;
}

