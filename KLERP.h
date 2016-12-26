/*******************************************************************
*   KLERP.h
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

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <future>
#include <immintrin.h>
#include <thread>

constexpr int32_t SHFT = 11;

void _KNERP(uint8_t* __restrict src, int32_t src_w, int32_t src_h, uint8_t* __restrict dst, int32_t dst_w, int32_t dst_h, int32_t start, int32_t stride, const int32_t* const __restrict x_idx) {
	// silliness to match opencv's output
	const double gys = 1.0 / (static_cast<double>(dst_h) / static_cast<double>(src_h));
	for (int32_t y = start; y < start + stride; ++y) {
		const uint8_t* __restrict const msrc = src + static_cast<int>(gys*y)*src_w;
		for (int32_t x = 0; x < dst_w; ++x) {
			dst[y*dst_w + x] = msrc[x_idx[x]];
		}
	}
}

template<bool multithread>
void KNERP(uint8_t* __restrict src, int32_t src_w, int32_t src_h, uint8_t* __restrict dst, int32_t dst_w, int32_t dst_h) {
	static const int32_t hw_concur = static_cast<int32_t>(std::thread::hardware_concurrency());
	static std::future<void>* const __restrict fut = new std::future<void>[hw_concur];

	// silliness to match opencv's output
	const double gxs = 1.0 / (static_cast<double>(dst_w) / static_cast<double>(src_w));
	int32_t* x_idx = reinterpret_cast<int32_t*>(malloc(dst_w * sizeof(int32_t)));
	for (int i = 0; i < dst_w; ++i) x_idx[i] = static_cast<int32_t>(gxs*i);

	if (!multithread || static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) < static_cast<size_t>(20000ULL)) {
		_KNERP(src, src_w, src_h, dst, dst_w, dst_h, 0, dst_h, x_idx);
	}
	else {
		const int stride = (dst_h - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(dst_h - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, _KNERP, src, src_w, src_h, dst, dst_w, dst_h, start, stride, x_idx);
		}
		fut[i] = std::async(std::launch::async, _KNERP, src, src_w, src_h, dst, dst_w, dst_h, start, dst_h - start, x_idx);
		for (int j = 0; j <= i; ++j) fut[j].wait();
	}
	free(x_idx);
}

void pass1(const uint8_t* src0, const uint8_t* src1, int32_t* interm0, int32_t* interm1, int32_t recycled, const int32_t* x_idx, const int16_t* x_wt, int32_t dst_w, int32_t x_last) {
	if (recycled == 0) {
		int32_t x = 0;
		for (; x < x_last; ++x) {
			const int32_t ix = x_idx[x];
			const int32_t wt0 = x_wt[2 * x];
			const int32_t wt1 = x_wt[2 * x + 1];
			const int32_t o0 = src0[ix] * wt0 + src0[ix + 1] * wt1;
			const int32_t o1 = src1[ix] * wt0 + src1[ix + 1] * wt1;
			interm0[x] = o0;
			interm1[x] = o1;
		}
		const int32_t ix = x_idx[x];
		const int32_t o0 = static_cast<int32_t>(src0[ix] << SHFT);
		const int32_t o1 = static_cast<int32_t>(src1[ix] << SHFT);
		for (; x < dst_w; ++x) {
			interm0[x] = o0;
			interm1[x] = o1;
		}
	}
	else {
		int32_t x = 0;
		for (; x < x_last; ++x) {
			const int32_t ix = x_idx[x];
			interm1[x] = src1[ix] * x_wt[2 * x] + src1[ix + 1] * x_wt[2 * x + 1];
		}

		const int32_t o = static_cast<int32_t>(src1[x_idx[x]] << SHFT);
		for (; x < dst_w; ++x) {
			interm1[x] = o;
		}
	}
}

void pass2(const int32_t* interm0, const int32_t* interm1, uint8_t* dst, const int16_t wt0, const int16_t wt1, int32_t dst_w) {
	const __m256i vwt0 = _mm256_set1_epi16(wt0);
	const __m256i vwt1 = _mm256_set1_epi16(wt1);
	constexpr short preshft = (1 << 2) / 2;
	const __m256i vpreshft = _mm256_set1_epi16(preshft);

	int32_t x = 0;
	for (; x <= dst_w - 32; x += 32) {
		__m256i a0 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm0 + x)), 4);
		__m256i a1 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm0 + x + 8)), 4);
		__m256i b0 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm1 + x)), 4);
		__m256i b1 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm1 + x + 8)), 4);
		// shifted 11 bits, then 7 bits.
		// a0,0:4 a1,0:4 a0,4:8 a1,4:8 because vpack* does not cross lane.
		a0 = _mm256_packs_epi32(a0, a1);
		b0 = _mm256_packs_epi32(b0, b1);

		__m256i a2 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm0 + x + 16)), 4);
		__m256i a3 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm0 + x + 24)), 4);
		__m256i b2 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm1 + x + 16)), 4);
		__m256i b3 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm1 + x + 24)), 4);
		a2 = _mm256_packs_epi32(a2, a3);
		b2 = _mm256_packs_epi32(b2, b3);

		a0 = _mm256_adds_epi16(_mm256_mulhi_epi16(a0, vwt0), _mm256_mulhi_epi16(b0, vwt1));
		// shifted 18 bits, then 2 bits after taking the high 16.
		a2 = _mm256_adds_epi16(_mm256_mulhi_epi16(a2, vwt0), _mm256_mulhi_epi16(b2, vwt1));

		a0 = _mm256_permute4x64_epi64(_mm256_srai_epi16(_mm256_adds_epi16(a0, vpreshft), 2), 216);
		// shifted 0 bits for output. Permute to cross lane.
		// Could also wait and permutevar8x32 once at the end instead. Roughly same speed.
		a2 = _mm256_permute4x64_epi64(_mm256_srai_epi16(_mm256_adds_epi16(a2, vpreshft), 2), 216);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), _mm256_permute4x64_epi64(_mm256_packus_epi16(a0, a2), 216));
	}

	for (; x <= dst_w - 16; x += 16) {
		__m256i a0 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm0 + x)), 4);
		__m256i a1 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm0 + x + 8)), 4);
		__m256i b0 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm1 + x)), 4);
		__m256i b1 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm1 + x + 8)), 4);
		a0 = _mm256_packs_epi32(a0, a1);
		b0 = _mm256_packs_epi32(b0, b1);

		a0 = _mm256_adds_epi16(_mm256_mulhi_epi16(a0, vwt0), _mm256_mulhi_epi16(b0, vwt1));
		a0 = _mm256_permute4x64_epi64(_mm256_srai_epi16(_mm256_adds_epi16(a0, vpreshft), 2), 216);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(a0, a0), 216)));
	}

	for (; x <= dst_w - 8; x += 8) {
		__m256i a0 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm0 + x)), 4);
		__m256i b0 = _mm256_srai_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(interm1 + x)), 4);
		a0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(a0, a0), 216);
		b0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(b0, b0), 216);
		a0 = _mm256_adds_epi16(_mm256_mulhi_epi16(a0, vwt0), _mm256_mulhi_epi16(b0, vwt1));
		a0 = _mm256_srai_epi16(_mm256_adds_epi16(a0, vpreshft), 2);
		a0 = _mm256_permute4x64_epi64(_mm256_packus_epi16(a0, a0), 216);
		*reinterpret_cast<int64_t*>(dst + x) = _mm_cvtsi128_si64(_mm256_castsi256_si128(a0));
	}

	for (; x <= dst_w - 4; x += 4) {
		__m128i a0 = _mm_srai_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(interm0 + x)), 4);
		__m128i b0 = _mm_srai_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(interm1 + x)), 4);
		a0 = _mm_packs_epi32(a0, a0);
		b0 = _mm_packs_epi32(b0, b0);
		a0 = _mm_adds_epi16(_mm_mulhi_epi16(a0, _mm256_castsi256_si128(vwt0)), _mm_mulhi_epi16(b0, _mm256_castsi256_si128(vwt1)));
		a0 = _mm_srai_epi16(_mm_adds_epi16(a0, _mm256_castsi256_si128(vpreshft)), 2);
		a0 = _mm_packus_epi16(a0, a0);
		*reinterpret_cast<int32_t*>(dst + x) = _mm_cvtsi128_si32(a0);
	}

	for (; x < dst_w; ++x) {
		dst[x] = static_cast<uint8_t>((((wt0 * (interm0[x] >> 4)) >> 16) + ((wt1 * (interm1[x] >> 4)) >> 16) + preshft) >> 2);
	}
}

void _KLERP(uint8_t* __restrict src, uint8_t* __restrict dst, const int32_t* x_idx, const int32_t* y_idx, const int16_t* x_wt, const int16_t* _y_wt, const int32_t src_w, const int32_t src_h, const int32_t dst_w, int32_t x_last, int32_t start, int32_t stride) {
	const int32_t x_pad = (dst_w + 31) & ~31;
	int32_t* interm = reinterpret_cast<int32_t*>(_mm_malloc(x_pad * 2 * sizeof(int32_t), 256));
	const uint8_t* src_rows[2] = { nullptr, nullptr };
	int32_t iy_last[2] = { -1, -1 };

	const int16_t* y_wt = _y_wt + start + start;

	for (int32_t y = start; y < start + stride; ++y, y_wt += 2) {
		int32_t iy0 = y_idx[y];
		int32_t iy = iy0 > 0 ? (iy0 < src_h ? iy0 : src_h - 1) : 0;

		int32_t recycle_rows = 0;
		if (iy == iy_last[0]) {
			recycle_rows = 2;
		}
		else if (iy == iy_last[1]) {
			memcpy(interm, interm + x_pad, x_pad * sizeof(int32_t));
			recycle_rows = 2;
		}
		src_rows[0] = src + src_w*iy;
		iy_last[0] = iy;

		++iy0;
		iy = iy0 > 0 ? (iy0 < src_h ? iy0 : src_h - 1) : 0;

		if (recycle_rows == 2 && iy != iy_last[1]) recycle_rows = 1;
		src_rows[1] = src + src_w*iy;
		iy_last[1] = iy;

		if (recycle_rows < 2) {
			pass1(src_rows[0], src_rows[1], interm, interm + x_pad, recycle_rows, x_idx, x_wt, dst_w, x_last);
		}
		pass2(interm, interm + x_pad, dst + dst_w*y, y_wt[0], y_wt[1], dst_w);
	}

	_mm_free(interm);
}

template<bool multithread>
void KLERP(uint8_t* __restrict src, int32_t src_w, int32_t src_h, uint8_t* __restrict dst, int32_t dst_w, int32_t dst_h) {
	static const int32_t hw_concur = static_cast<int32_t>(std::thread::hardware_concurrency());
	static std::future<void>* const __restrict fut = new std::future<void>[hw_concur];

	// silliness to match opencv's output
	const double gxs = 1.0 / (static_cast<double>(dst_w) / static_cast<double>(src_w));
	const double gys = 1.0 / (static_cast<double>(dst_h) / static_cast<double>(src_h));

	int32_t* x_idx = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t)*(dst_w + dst_h) + 2 * sizeof(int16_t)*(dst_w + dst_h)));
	int32_t* y_idx = x_idx + dst_w;
	int16_t* x_wt = reinterpret_cast<int16_t*>(y_idx + dst_h);
	int16_t* iy_wt = x_wt + dst_w * 2;
	int32_t x_last = dst_w;

	for (int32_t x = 0; x < dst_w; ++x) {
		float fx = static_cast<float>((x + 0.5)*gxs - 0.5);
		int32_t ix = _mm_cvtss_si32(_mm_round_ps(_mm_set1_ps(fx), _MM_FROUND_TO_NEG_INF));
		fx -= ix;

		if (ix < 0) {
			fx = 0.0f;
			ix = 0;
		}

		if (ix >= src_w - 1) {
			if (x_last > x) x_last = x;
			fx = 0.0f;
			ix = src_w - 1;
		}

		x_idx[x] = ix;
		x_wt[2 * x] = static_cast<int16_t>(_mm_cvtss_si32(_mm_set_ss((1.0f - fx) * (1 << SHFT))));
		x_wt[2 * x + 1] = static_cast<int16_t>(_mm_cvtss_si32(_mm_set_ss(fx * (1 << SHFT))));
	}

	for (int32_t y = 0; y < dst_h; ++y) {
		float fy = static_cast<float>((y + 0.5)*gys - 0.5);
		int32_t iy = _mm_cvtss_si32(_mm_round_ps(_mm_set1_ps(fy), _MM_FROUND_TO_NEG_INF));
		fy -= iy;

		y_idx[y] = iy;
		iy_wt[2 * y] = static_cast<int16_t>(_mm_cvtss_si32(_mm_set_ss((1.0f - fy) * (1 << SHFT))));
		iy_wt[2 * y + 1] = static_cast<int16_t>(_mm_cvtss_si32(_mm_set_ss(fy * (1 << SHFT))));
	}

	if (!multithread || static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) < static_cast<size_t>(20000ULL)) {
		_KLERP(src, dst, x_idx, y_idx, x_wt, iy_wt, src_w, src_h, dst_w, x_last, 0, dst_h);
	}
	else {
		const int32_t stride = (dst_h - 1) / hw_concur + 1;
		int32_t i = 0;
		int32_t start = 0;
		for (; i < std::min(dst_h - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, _KLERP, src, dst, x_idx, y_idx, x_wt, iy_wt, src_w, src_h, dst_w, x_last, start, stride);
		}
		fut[i] = std::async(std::launch::async, _KLERP, src, dst, x_idx, y_idx, x_wt, iy_wt, src_w, src_h, dst_w, x_last, start, dst_h - start);
		for (int32_t j = 0; j <= i; ++j) fut[j].wait();
	}

	free(x_idx);
}