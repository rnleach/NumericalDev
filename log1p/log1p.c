#include "../tests/eagle_test.h"

static inline f64
eagle_log1p(f64 x)
{

    if(eagle_isnan(x) | (x < -1.0)) { return EAGLE_NAN; }
    if(x == -1.0) { return EAGLE_NEG_INF; }
    if(x == EAGLE_POS_INF) { return EAGLE_POS_INF; }

    f64 const small_x_range_low = -0x1.2bec333018867p-2;
    f64 const small_x_range_high = 0x1.a827999fcef32p-2;

    if((x >= small_x_range_low) & (x <= small_x_range_high))
    {

        f64 const c00 = -0x1p-1;
        f64 const c01 = 0x1.555555555555dp-2;
        f64 const c02 = -0x1.fffffffffff96p-3;
        f64 const c03 = 0x1.9999999997539p-3;
        f64 const c04 = -0x1.555555555cabap-3;
        f64 const c05 = 0x1.249249262df8ap-3;
        f64 const c06 = -0x1.fffffffbce703p-4;
        f64 const c07 = 0x1.c71c70bc068a6p-4;
        f64 const c08 = -0x1.9999999c93ea6p-4;
        f64 const c09 = 0x1.745d4634b62aep-4;
        f64 const c10 = -0x1.5555832c1a87ap-4;
        f64 const c11 = 0x1.3b0eeabfe4366p-4;
        f64 const c12 = -0x1.2489989b9a832p-4;
        f64 const c13 = 0x1.1159011b4deb6p-4;
        f64 const c14 = -0x1.00c09795798c4p-4;
        f64 const c15 = 0x1.dd75dae546896p-5;
        f64 const c16 = -0x1.b59d93cf0b7fdp-5;
        f64 const c17 = 0x1.b9138d908b63bp-5;
        f64 const c18 = -0x1.fcf84fe53897bp-5;
        f64 const c19 = 0x1.d44165bff8e26p-5;
        f64 const c20 = -0x1.978d46ca90ecbp-6;

        f64 const x2  = x * x;
        f64 const x4  = x2 * x2;
        f64 const x8  = x4 * x4;
        f64 const x16 = x8 * x8;

        f64 const t0  = __builtin_fma(c01, x, c00);
        f64 const t1  = __builtin_fma(c03, x, c02);
        f64 const t2  = __builtin_fma(c05, x, c04);
        f64 const t3  = __builtin_fma(c07, x, c06);
        f64 const t4  = __builtin_fma(c09, x, c08);
        f64 const t5  = __builtin_fma(c11, x, c10);
        f64 const t6  = __builtin_fma(c13, x, c12);
        f64 const t7  = __builtin_fma(c15, x, c14);
        f64 const t8  = __builtin_fma(c17, x, c16);
        f64 const t9  = __builtin_fma(c19, x, c18);
        f64 const t10 = c20;

        f64 const s0 = __builtin_fma(t1, x2, t0);   /* 0–3              */
        f64 const s1 = __builtin_fma(t3, x2, t2);   /* 4–7              */
        f64 const s2 = __builtin_fma(t5, x2, t4);   /* 8–11             */
        f64 const s3 = __builtin_fma(t7, x2, t6);   /* 12–15            */
        f64 const s4 = __builtin_fma(t9, x2, t8);   /* 16–19            */

        f64 const u0 = __builtin_fma(s1, x4, s0);   /* 0–7              */
        f64 const u1 = __builtin_fma(s3, x4, s2);   /* 8–15             */

        f64 p = __builtin_fma(u1, x8, u0);          /* 0–15             */
        f64 const v = __builtin_fma(t10, x4, s4);   /* 16–20 (relative) */
        p = __builtin_fma(v, x16, p);

        return __builtin_fma(p, x2, x);
    }
    else
    {
        f64 s_hi, s_lo;
        eagle_twosum(1.0, x, &s_hi, &s_lo);

        f64 corr = s_lo / s_hi;
        return eagle_log(s_hi) + corr;
    }
}

#ifdef __AVX2__

static inline __m256d
eagle_avx2_log1p_pd(__m256d x)
{
    __m256d isnan_mask = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);
    __m256d const lt_neg_one_mask = _mm256_cmp_pd(x, _mm256_set1_pd(-1.0), _CMP_LT_OQ);
    isnan_mask = _mm256_or_pd(isnan_mask, lt_neg_one_mask);

    __m256d const neg_one_mask = _mm256_cmp_pd(x, _mm256_set1_pd(-1.0), _CMP_EQ_OQ);
    __m256d const posinf_mask  = _mm256_cmp_pd(x, _mm256_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);

    __m256d const small_x_range_low = _mm256_cmp_pd(x, _mm256_set1_pd(-0x1.2bec333018867p-2), _CMP_GE_OQ);
    __m256d const small_x_range_high = _mm256_cmp_pd(x, _mm256_set1_pd(0x1.a827999fcef32p-2), _CMP_LE_OQ);
    __m256d const small_mask = _mm256_and_pd(small_x_range_low, small_x_range_high);

    /* Handle small x */
    __m256d small_x_result;
    {
        __m256d const c00 = _mm256_set1_pd(-0x1p-1);
        __m256d const c01 = _mm256_set1_pd(0x1.555555555555dp-2);
        __m256d const c02 = _mm256_set1_pd(-0x1.fffffffffff96p-3);
        __m256d const c03 = _mm256_set1_pd(0x1.9999999997539p-3);
        __m256d const c04 = _mm256_set1_pd(-0x1.555555555cabap-3);
        __m256d const c05 = _mm256_set1_pd(0x1.249249262df8ap-3);
        __m256d const c06 = _mm256_set1_pd(-0x1.fffffffbce703p-4);
        __m256d const c07 = _mm256_set1_pd(0x1.c71c70bc068a6p-4);
        __m256d const c08 = _mm256_set1_pd(-0x1.9999999c93ea6p-4);
        __m256d const c09 = _mm256_set1_pd(0x1.745d4634b62aep-4);
        __m256d const c10 = _mm256_set1_pd(-0x1.5555832c1a87ap-4);
        __m256d const c11 = _mm256_set1_pd(0x1.3b0eeabfe4366p-4);
        __m256d const c12 = _mm256_set1_pd(-0x1.2489989b9a832p-4);
        __m256d const c13 = _mm256_set1_pd(0x1.1159011b4deb6p-4);
        __m256d const c14 = _mm256_set1_pd(-0x1.00c09795798c4p-4);
        __m256d const c15 = _mm256_set1_pd(0x1.dd75dae546896p-5);
        __m256d const c16 = _mm256_set1_pd(-0x1.b59d93cf0b7fdp-5);
        __m256d const c17 = _mm256_set1_pd(0x1.b9138d908b63bp-5);
        __m256d const c18 = _mm256_set1_pd(-0x1.fcf84fe53897bp-5);
        __m256d const c19 = _mm256_set1_pd(0x1.d44165bff8e26p-5);
        __m256d const c20 = _mm256_set1_pd(-0x1.978d46ca90ecbp-6);

        __m256d const x2  = _mm256_mul_pd(x, x);
        __m256d const x4  = _mm256_mul_pd(x2, x2);
        __m256d const x8  = _mm256_mul_pd(x4, x4);
        __m256d const x16 = _mm256_mul_pd(x8, x8);

        __m256d const t0  = _mm256_fmadd_pd(c01, x, c00);
        __m256d const t1  = _mm256_fmadd_pd(c03, x, c02);
        __m256d const t2  = _mm256_fmadd_pd(c05, x, c04);
        __m256d const t3  = _mm256_fmadd_pd(c07, x, c06);
        __m256d const t4  = _mm256_fmadd_pd(c09, x, c08);
        __m256d const t5  = _mm256_fmadd_pd(c11, x, c10);
        __m256d const t6  = _mm256_fmadd_pd(c13, x, c12);
        __m256d const t7  = _mm256_fmadd_pd(c15, x, c14);
        __m256d const t8  = _mm256_fmadd_pd(c17, x, c16);
        __m256d const t9  = _mm256_fmadd_pd(c19, x, c18);
        __m256d const t10 = c20;

        __m256d const s0 = _mm256_fmadd_pd(t1, x2, t0);   /* 0–3              */
        __m256d const s1 = _mm256_fmadd_pd(t3, x2, t2);   /* 4–7              */
        __m256d const s2 = _mm256_fmadd_pd(t5, x2, t4);   /* 8–11             */
        __m256d const s3 = _mm256_fmadd_pd(t7, x2, t6);   /* 12–15            */
        __m256d const s4 = _mm256_fmadd_pd(t9, x2, t8);   /* 16–19            */

        __m256d const u0 = _mm256_fmadd_pd(s1, x4, s0);   /* 0–7              */
        __m256d const u1 = _mm256_fmadd_pd(s3, x4, s2);   /* 8–15             */

        __m256d p = _mm256_fmadd_pd(u1, x8, u0);          /* 0–15             */
        __m256d const v = _mm256_fmadd_pd(t10, x4, s4);   /* 16–20 (relative) */
        p = _mm256_fmadd_pd(v, x16, p);

        small_x_result = _mm256_fmadd_pd(p, x2, x);
    }

    /* Handle NOT small x */
    __m256d not_small_x_result;
    {
        __m256d s_hi, s_lo;
        eagle_avx2_twosum(_mm256_set1_pd(1.0), x, &s_hi, &s_lo);

        __m256d corr = _mm256_div_pd(s_lo, s_hi);
        not_small_x_result = _mm256_add_pd(eagle_avx2_log_pd(s_hi), corr);
    }

    __m256d final = _mm256_blendv_pd(not_small_x_result, small_x_result, small_mask);

    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_NAN), isnan_mask);
    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_POS_INF), posinf_mask);
    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_NEG_INF), neg_one_mask);

    return final;

}

#endif

#if ELK_AVX_512

static inline __m512d
eagle_avx512_log1p_pd(__m512d x)
{
    __mmask8 isnan_mask = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
    __mmask8 const lt_neg_one_mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(-1.0), _CMP_LT_OQ);
    isnan_mask = isnan_mask | lt_neg_one_mask;

    __mmask8 const neg_one_mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(-1.0), _CMP_EQ_OQ);
    __mmask8 const posinf_mask  = _mm512_cmp_pd_mask(x, _mm512_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);

    __mmask8 const small_x_range_low = _mm512_cmp_pd_mask(x, _mm512_set1_pd(-0x1.2bec333018867p-2), _CMP_GE_OQ);
    __mmask8 const small_x_range_high = _mm512_cmp_pd_mask(x, _mm512_set1_pd(0x1.a827999fcef32p-2), _CMP_LE_OQ);
    __mmask8 const small_mask = small_x_range_low & small_x_range_high;

    /* Handle small x */
    __m512d small_x_result;
    {
        __m512d const c00 = _mm512_set1_pd(-0x1p-1);
        __m512d const c01 = _mm512_set1_pd(0x1.555555555555dp-2);
        __m512d const c02 = _mm512_set1_pd(-0x1.fffffffffff96p-3);
        __m512d const c03 = _mm512_set1_pd(0x1.9999999997539p-3);
        __m512d const c04 = _mm512_set1_pd(-0x1.555555555cabap-3);
        __m512d const c05 = _mm512_set1_pd(0x1.249249262df8ap-3);
        __m512d const c06 = _mm512_set1_pd(-0x1.fffffffbce703p-4);
        __m512d const c07 = _mm512_set1_pd(0x1.c71c70bc068a6p-4);
        __m512d const c08 = _mm512_set1_pd(-0x1.9999999c93ea6p-4);
        __m512d const c09 = _mm512_set1_pd(0x1.745d4634b62aep-4);
        __m512d const c10 = _mm512_set1_pd(-0x1.5555832c1a87ap-4);
        __m512d const c11 = _mm512_set1_pd(0x1.3b0eeabfe4366p-4);
        __m512d const c12 = _mm512_set1_pd(-0x1.2489989b9a832p-4);
        __m512d const c13 = _mm512_set1_pd(0x1.1159011b4deb6p-4);
        __m512d const c14 = _mm512_set1_pd(-0x1.00c09795798c4p-4);
        __m512d const c15 = _mm512_set1_pd(0x1.dd75dae546896p-5);
        __m512d const c16 = _mm512_set1_pd(-0x1.b59d93cf0b7fdp-5);
        __m512d const c17 = _mm512_set1_pd(0x1.b9138d908b63bp-5);
        __m512d const c18 = _mm512_set1_pd(-0x1.fcf84fe53897bp-5);
        __m512d const c19 = _mm512_set1_pd(0x1.d44165bff8e26p-5);
        __m512d const c20 = _mm512_set1_pd(-0x1.978d46ca90ecbp-6);

        __m512d const x2  = _mm512_mul_pd(x, x);
        __m512d const x4  = _mm512_mul_pd(x2, x2);
        __m512d const x8  = _mm512_mul_pd(x4, x4);
        __m512d const x16 = _mm512_mul_pd(x8, x8);

        __m512d const t0  = _mm512_fmadd_pd(c01, x, c00);
        __m512d const t1  = _mm512_fmadd_pd(c03, x, c02);
        __m512d const t2  = _mm512_fmadd_pd(c05, x, c04);
        __m512d const t3  = _mm512_fmadd_pd(c07, x, c06);
        __m512d const t4  = _mm512_fmadd_pd(c09, x, c08);
        __m512d const t5  = _mm512_fmadd_pd(c11, x, c10);
        __m512d const t6  = _mm512_fmadd_pd(c13, x, c12);
        __m512d const t7  = _mm512_fmadd_pd(c15, x, c14);
        __m512d const t8  = _mm512_fmadd_pd(c17, x, c16);
        __m512d const t9  = _mm512_fmadd_pd(c19, x, c18);
        __m512d const t10 = c20;

        __m512d const s0 = _mm512_fmadd_pd(t1, x2, t0);   /* 0–3              */
        __m512d const s1 = _mm512_fmadd_pd(t3, x2, t2);   /* 4–7              */
        __m512d const s2 = _mm512_fmadd_pd(t5, x2, t4);   /* 8–11             */
        __m512d const s3 = _mm512_fmadd_pd(t7, x2, t6);   /* 12–15            */
        __m512d const s4 = _mm512_fmadd_pd(t9, x2, t8);   /* 16–19            */

        __m512d const u0 = _mm512_fmadd_pd(s1, x4, s0);   /* 0–7              */
        __m512d const u1 = _mm512_fmadd_pd(s3, x4, s2);   /* 8–15             */

        __m512d p = _mm512_fmadd_pd(u1, x8, u0);          /* 0–15             */
        __m512d const v = _mm512_fmadd_pd(t10, x4, s4);   /* 16–20 (relative) */
        p = _mm512_fmadd_pd(v, x16, p);

        small_x_result = _mm512_fmadd_pd(p, x2, x);
    }

    /* Handle NOT small x */
    __m512d not_small_x_result;
    {
        __m512d s_hi, s_lo;
        eagle_avx512_twosum(_mm512_set1_pd(1.0), x, &s_hi, &s_lo);

        __m512d corr = _mm512_div_pd(s_lo, s_hi);
        not_small_x_result = _mm512_add_pd(eagle_avx512_log_pd(s_hi), corr);
    }

    __m512d final = _mm512_mask_blend_pd(small_mask, not_small_x_result, small_x_result);

    final = _mm512_mask_blend_pd(isnan_mask, final, _mm512_set1_pd(EAGLE_NAN));
    final = _mm512_mask_blend_pd(posinf_mask, final, _mm512_set1_pd(EAGLE_POS_INF));
    final = _mm512_mask_blend_pd(neg_one_mask, final, _mm512_set1_pd(EAGLE_NEG_INF));

    return final;
}

#endif 

