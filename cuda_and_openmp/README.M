CUDA:

Anaglyph:

/usr/local/cuda/bin/nvcc image_anaglyph.cu `pkg-config opencv4 --cflags --libs` imagecpp_anaglyph.cpp -o imagecuda_a

./imagecuda_a flower_resized.jpg 2


Image processing:

/usr/local/cuda/bin/nvcc image_processing.cu `pkg-config opencv4 --cflags --libs` imagecpp_image_processing.cpp -o imagecuda_ip

./imagecuda_ip flower_resized.jpg 3 1.0


Denoising:

/usr/local/cuda/bin/nvcc image_denoising.cu `pkg-config opencv4 --cflags --libs` imagecpp_denoising.cpp -o imagecuda_dn

./imagecuda_dn flower_resized.jpg 3 1.0


OpenMP:

g++ opencv-omp-anaglyph.cpp -fopenmp `pkg-config opencv4 --cflags` -c

g++ opencv-omp-anaglyph.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-anaglyph

./opencv-omp-anaglyph flower_resized.jpg 2


g++ opencv-omp-processing.cpp -fopenmp `pkg-config opencv4 --cflags` -c

g++ opencv-omp-processing.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-processing

./opencv-omp-processing flower_resized.jpg 3 1.0


g++ opencv-omp-denoising.cpp -fopenmp `pkg-config opencv4 --cflags` -c

g++ opencv-omp-denoising.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o opencv-omp-denoising

./opencv-omp-denoising flower_resized.jpg 3 10000000.0

