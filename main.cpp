/*******************************************************************
*   main.cpp
*   KLERP
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Dec 26, 2016
*******************************************************************/
// 
// The file KLERP.h exposes two extremely high performance CPU
// resize operations,
// KLERP (bilinear interpolation), and
// KNERP (nearest neighbor interpolation),
// both of which use multithreading and various
// scalar optimizations. KLERP is also written partly in AVX2
// so an AVX2-ready CPU is required.
//
// These are state-of-the-art CPU-side interpolators, exceeding
// OpenCV's implementation in speed by 25-100% while 
// matching its output and capabilities in both interpolation modes.
//
// The magnitude of the speed gain depends on whether
// you're downscaling or upscaling and by how much. 
//
// For even more speed, see my CUDA version, CUDALERP, at
// https://github.com/komrad36/CUDALERP.
//
// All functionality is contained in the header 'KLERP.h'
// and has no external dependencies at all.
//
// Note that these are intended for computer vision use
// (hence the speed) and are designed for grayscale images.
//
// The file 'main.cpp' is an example and speed test driver.
// 

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "KLERP.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

int main() {
	constexpr auto warmups = 2000;
	constexpr auto runs = 2000;
	constexpr bool multithreading = true;

	auto image = new uint8_t[4];
	image[0] = 255;
	image[1] = 255;
	image[2] = 0;
	image[3] = 0;

	constexpr int oldw = 2;
	constexpr int oldh = 2;
	constexpr int neww = static_cast<int>(static_cast<double>(oldw) * 400.0);
	constexpr int newh = static_cast<int>(static_cast<double>(oldh) * 1000.0);

	auto out = new uint8_t[neww * newh];

	for (int i = 0; i < warmups; ++i) {
		KLERP<multithreading>(image, oldw, oldh, out, neww, newh);
	}

	auto start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) {
		KLERP<multithreading>(image, oldw, oldh, out, neww, newh);
	}
	auto end = high_resolution_clock::now();
	auto sum = (end - start) / runs;

	std::cout << "KLERP took " << static_cast<double>(sum.count()) * 1e-3 << " us." << std::endl;

	std::cout << "Input stats: " << oldh << " rows, " << oldw << " cols." << std::endl;
	std::cout << "Output stats: " << newh << " rows, " << neww << " cols." << std::endl;
}
