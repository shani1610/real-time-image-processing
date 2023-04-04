echo CUDA:

echo CUDA Anaglyph:

/usr/local/cuda/bin/nvcc image_anaglyph.cu `pkg-config opencv4 --cflags --libs` imagecpp_anaglyph.cpp -o imagecuda_a

echo CUDA Image processing:

/usr/local/cuda/bin/nvcc image_processing.cu `pkg-config opencv4 --cflags --libs` imagecpp_processing.cpp -o imagecuda_ip

echo CUDA Denoising:

/usr/local/cuda/bin/nvcc image_denoising.cu `pkg-config opencv4 --cflags --libs` imagecpp_denoising.cpp -o imagecuda_dn

echo OpenMP:

echo OpenMP Anaglyph:

g++ opencv-omp-anaglyph.cpp -fopenmp `pkg-config opencv4 --cflags` -c

g++ opencv-omp-anaglyph.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-anaglyph

echo OpenMP Image processing: 

g++ opencv-omp-processing.cpp -fopenmp `pkg-config opencv4 --cflags` -c

g++ opencv-omp-processing.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-processing

echo OpenMP Denoising:

g++ opencv-omp-denoising.cpp -fopenmp `pkg-config opencv4 --cflags` -c

g++ opencv-omp-denoising.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-denoising

