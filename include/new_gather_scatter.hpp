
#ifndef NEW_GATHER_SCATTER_HPP
#define NEW_GATHER_SCATTER_HPP

#include <immintrin.h>
#include "xsimd/xsimd.hpp"

namespace xsimd
{
    template<>
    template<>
    XSIMD_INLINE batch<double, avx2> batch<double, avx2>::gather<double, int32_t>(double const* src, batch<int32_t, avx2> const& index) noexcept
    {
        return _mm256_i32gather_pd(src, _mm256_castsi256_si128(index), sizeof(double));
    }

    template<>
    template<>
    XSIMD_INLINE batch<double, avx512f> batch<double, avx512f>::gather<double, int32_t>(double const* src, batch<int32_t, avx512f> const& index) noexcept
    {
        return _mm512_i32gather_pd(_mm512_castsi512_si256(index), src, sizeof(double));
    }

    template<>
    template<>
    XSIMD_INLINE void batch<double, avx2>::scatter<double, int32_t>(double* dst, batch<int32_t, avx2> const& index) const noexcept
    {
        _mm256_i32scatter_pd(dst, _mm256_castsi256_si128(index), *this, sizeof(double));
    }

    template<>
    template<>
    XSIMD_INLINE void batch<double, avx512f>::scatter<double, int32_t>(double* dst, batch<int32_t, avx512f> const& index) const noexcept
    {
        _mm512_i32scatter_pd(dst, _mm512_castsi512_si256(index), *this, sizeof(double));
    }
}

#endif 