# Real Time Image Processing

This repository contains the results of practical works for the Real time processing of conventional and non-conventional images with
GPUs class in Jean Monnet University, lectured by Professor Philippe Colantoni. 
This course introduces basic and advanced techniques dedicated for General-Purpose processing on
Graphics Processing Unit (GPGPU). It introduces the basic concepts of parallel programming and shows how
to use the computing power of modern GPUs for conventional/non-conventional images processing in real
time. 

## Content
Introduction to parallel programming; Introduction to General-purpose processing on graphics processing
units (GPGPU): GPGPU with shaders, CUDA; Image processing with graphic shaders and compute shaders
(application with WebGL for web applications); CUDA based image processing (application with OpenCV for
native applications)
Case of studies: Implementation of conventional color image/video processing methods, Implementation of
non-conventional image processing methods

| Path | Description
| :--- | :----------
| [Shaders Based Image Processing](https://shani1610.github.io/real-time-image-processing/shader_based_image_processing/) | Practical Work 1
| [OpenMP and CUDA Based Image Processing](https://github.com/shani1610/real-time-image-processing/tree/main/cuda_and_openmp/) | Practical Work 2

### Practical Work 1: Shaders Based Image Processing


### Practical Work 2: OpenMP and CUDA Based Image Processing
run from the directory cuda_and_openmp

``` sh compile.sh ```

if you want to run only for one method, you can see the commands for each method inside the compile.sh.

the implemented anaglyph methods equations were taken from this [link](https://3dtv.at/Knowhow/AnaglyphComparison_en.aspx).

| Method Name | Method Number
| :--- | :----------
| True | 0
| Gray | 1
| Color | 2
| Half-Color | 3
| Optimized | 4


to execute run these: 

#### CUDA 

Anaglyph

```./imagecuda_a <<image path>> <<method number>>```

Example: ```./imagecuda_a flower_resized.jpg 2```

Gaussian 

```./imagecuda_ip <<image path>> <<kernel size divided by 2>> <<sigma>>```

Example: ```./imagecuda_ip flower_resized.jpg 3 1.0```

Denoising 

```./imagecuda_dn <<image path>> <<neighborhood size for the covariance matrix divided by 2>> <<factor ratio applied to determine the gaussian kernel size>>```

Example: ```./imagecuda_dn flower_resized.jpg 3 1.0```

#### OpenMP

Anaglyph

```./opencv-omp-anaglyph flower_resized.jpg 2```

Gaussian 

```./opencv-omp-processing flower_resized.jpg 3 1.0```

Denoising 

```./opencv-omp-denoising flower_resized.jpg 3 10000000.0```


### Acknowledgements

parts of the code borrowed from code given in class.






