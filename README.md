The file KLERP.h exposes two extremely high performance CPU
resize operations,
KLERP (bilinear interpolation), and
KNERP (nearest neighbor interpolation),
both of which use multithreading and various
scalar optimizations. KLERP is also written partly in AVX2
so an AVX2-ready CPU is required.

These are state-of-the-art CPU-side interpolators, exceeding
OpenCV's implementation in speed by 25-100% while 
matching its output and capabilities in both interpolation modes.

The magnitude of the speed gain depends on whether
you're downscaling or upscaling and by how much. 

For even more speed, see my CUDA version, CUDALERP, at
https://github.com/komrad36/CUDALERP.

All functionality is contained in the header 'KLERP.h'
and has no external dependencies at all.

Note that these are intended for computer vision use
(hence the speed) and are designed for grayscale images.

The file 'main.cpp' is an example and speed test driver.
