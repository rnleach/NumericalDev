#include "../tests/eagle_test.h"

/* log(1.0 - exp(-x)) */

static inline f64
eagle_log1mexp(f64 x)
{

    if(eagle_isnan(x) | (x < 0.0)) { return EAGLE_NAN; }
    if(x == 0.0) { return EAGLE_NEG_INF; }
    if(x == EAGLE_POS_INF) { return 0.0; }

    f64 const ln2 = 0x1.62e42fefa39efp-1;   /* ln(2) */

    if(x <= ln2)
    {
        f64 const y = x < 0x1p-52 ? x : -eagle_expm1(-x);
        return eagle_log(y);
    }
    else
    {
        return eagle_log1p(-eagle_exp(-x));
    }
}

#ifdef __AVX2__

static inline __m256d
eagle_avx2_log1mexp_pd(__m256d x)
{
    __m256d final = _mm256_set1_pd(EAGLE_NAN);

    __m256d const ln2 = _mm256_set1_pd(0x1.62e42fefa39efp-1);
    __m256d const smalls_mask = _mm256_cmp_pd(x, ln2, _CMP_LE_OQ);
    i32 smalls_exist = _mm256_movemask_pd(smalls_mask);
    i32 larges_exist = (~smalls_exist) & 0xF;

    __m256d const neg_x = eagle_avx2_negate_pd(x);

    /* Handle large x */
    if(larges_exist)
    {
        __m256d y2 = eagle_avx2_exp_pd(neg_x);
        y2 = eagle_avx2_negate_pd(y2);
        __m256d large_x_result = eagle_avx2_log1p_pd(y2);
        final = large_x_result;
    }

    /* Handle small x */
    if(smalls_exist)
    {
        __m256d const tiny = _mm256_set1_pd(0x1p-52);
        __m256d const tiny_mask = _mm256_cmp_pd(x, tiny, _CMP_LT_OQ);
        __m256d y1 = eagle_avx2_expm1_pd(neg_x);
        y1 = eagle_avx2_negate_pd(y1);
        y1 = _mm256_blendv_pd(y1, x, tiny_mask);
        __m256d small_x_result = eagle_avx2_log_pd(y1);

        final = _mm256_blendv_pd(final, small_x_result, smalls_mask);
    }

    return final;
}

#endif

#if ELK_AVX_512

static inline __m512d
eagle_avx512_log1mexp_pd(__m512d x)
{
    __m512d final = _mm512_set1_pd(EAGLE_NAN);

    __m512d const ln2 = _mm512_set1_pd(0x1.62e42fefa39efp-1);
    __mmask8 const smalls_mask = _mm512_cmp_pd_mask(x, ln2, _CMP_LE_OQ);
    __mmask8 const larges_mask = (~smalls_mask) & 0xFF;

    __m512d const neg_x = eagle_avx512_negate_pd(x);

    /* Handle NOT small x */
    if(larges_mask)
    {
        __m512d y = eagle_avx512_exp_pd(neg_x);
        y = eagle_avx512_negate_pd(y);
        __m512d large_x_result = eagle_avx512_log1p_pd(y);

        final = large_x_result;
    }

    /* Handle small x */
    if(smalls_mask)
    {
        __m512d const tiny = _mm512_set1_pd(0x1p-52);
        __mmask8 const tiny_mask = _mm512_cmp_pd_mask(x, tiny, _CMP_LT_OQ);
        __m512d y = eagle_avx512_expm1_pd(neg_x);
        y = eagle_avx512_negate_pd(y);
        y = _mm512_mask_blend_pd(tiny_mask, y, x);
        __m512d small_x_result = eagle_avx512_log_pd(y);

        final = _mm512_mask_blend_pd(smalls_mask, final, small_x_result);
    }

    return final;
}

#endif 

