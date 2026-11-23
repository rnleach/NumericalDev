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
    /* TODO: Inline functions above and do the masking below only once? For a performance boost? */

    /*
    __m256d isnan_mask = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);
    __m256d const lt_zero_mask = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_LT_OQ);
    isnan_mask = _mm256_or_pd(isnan_mask, lt_zero_mask);

    __m256d const zero_mask = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_EQ_OQ);
    __m256d const posinf_mask  = _mm256_cmp_pd(x, _mm256_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);

    */

    __m256d const ln2 = _mm256_set1_pd(0x1.62e42fefa39efp-1);
    __m256d const tiny = _mm256_set1_pd(0x1p-52);
    __m256d const neg_x = eagle_avx2_negate_pd(x);
    
    /* Handle small x */
    __m256d const small_mask = _mm256_cmp_pd(x, ln2, _CMP_LE_OQ);
    __m256d const tiny_mask = _mm256_cmp_pd(x, tiny, _CMP_LT_OQ);
    __m256d small_x_result;
    {
        __m256d y1 = eagle_avx2_expm1_pd(neg_x);
        y1 = eagle_avx2_negate_pd(y1);
        y1 = _mm256_blendv_pd(y1, x, tiny_mask);
        small_x_result = eagle_avx2_log_pd(y1);
    }

    /* Handle large x */
    __m256d large_x_result;
    {
        __m256d y2 = eagle_avx2_exp_pd(neg_x);
        y2 = eagle_avx2_negate_pd(y2);
        large_x_result = eagle_avx2_log1p_pd(y2);
    }

    __m256d final = _mm256_blendv_pd(large_x_result, small_x_result, small_mask);

    /* This is not necessary because it was handled in the large and small cases above
    
    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_NAN), isnan_mask);
    final = _mm256_blendv_pd(final, _mm256_set1_pd(0.0), posinf_mask);
    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_NEG_INF), zero_mask);

    */

    return final;
}

#endif

#if ELK_AVX_512

static inline __m512d
eagle_avx512_log1mexp_pd(__m512d x)
{
    /* TODO: Inline functions and do the masking below only once? For a performance boost? */

    /*
    __mmask8 isnan_mask = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
    __mmask8 const lt_zero_mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(0.0), _CMP_LT_OQ);
    isnan_mask = isnan_mask | lt_zero_mask;

    __mmask8 const zero_mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(0.0), _CMP_EQ_OQ);
    __mmask8 const posinf_mask  = _mm512_cmp_pd_mask(x, _mm512_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);

    */

    __m512d const ln2 = _mm512_set1_pd(0x1.62e42fefa39efp-1);
    __m512d const tiny = _mm512_set1_pd(0x1p-52);
    __m512d const neg_x = eagle_avx512_negate_pd(x);

    /* Handle small x */
    __mmask8 const small_mask = _mm512_cmp_pd_mask(x, ln2, _CMP_LE_OQ);
    __mmask8 const tiny_mask = _mm512_cmp_pd_mask(x, tiny, _CMP_LT_OQ);
    __m512d small_x_result;
    {
        __m512d y = eagle_avx512_expm1_pd(neg_x);
        y = eagle_avx512_negate_pd(y);
        y = _mm512_mask_blend_pd(tiny_mask, y, x);
        small_x_result = eagle_avx512_log_pd(y);
    }

    /* Handle NOT small x */
    __m512d large_x_result;
    {
        __m512d y = eagle_avx512_exp_pd(neg_x);
        y = eagle_avx512_negate_pd(y);
        large_x_result = eagle_avx512_log1p_pd(y);
    }

    __m512d final = _mm512_mask_blend_pd(small_mask, large_x_result, small_x_result);

    /* This is not necessary because it was handled in the large and small cases above
    
    final = _mm512_mask_blend_pd(isnan_mask, final, _mm512_set1_pd(EAGLE_NAN));
    final = _mm512_mask_blend_pd(posinf_mask, final, _mm512_set1_pd(0.0));
    final = _mm512_mask_blend_pd(zero_mask, final, _mm512_set1_pd(EAGLE_NEG_INF));

    */

    return final;
}

#endif 

