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

        f64 const c00l = -0x1p-1;
        f64 const c01l = 0x1.5555555555165p-2;
        f64 const c02l = -0x1.00000000013a6p-2;
        f64 const c03l = 0x1.999999b4d68bdp-3;
        f64 const c04l = -0x1.55554ce5cc951p-3;
        f64 const c05l = 0x1.249375d994fa6p-3;
        f64 const c06l = -0x1.ffcf7eb545b5cp-4;
        f64 const c07l = 0x1.c99c47194fe02p-4;
        f64 const c08l = -0x1.83879be1f1843p-4;
        f64 const c09l = 0x1.f949fac831c28p-4;
        f64 const c10l = 0x1.9562af4786612p-5;
        f64 const c11l = 0x1.bdcd703e7e36dp-2;
        f64 const c12l = 0x1.0a0fd969850b5p-1;
        f64 const c13l = 0x1.1baa608cdc3abp-1;

        f64 const c00h = -0x1.fffffffffffcap-2;
        f64 const c01h = 0x1.5555555552045p-2;
        f64 const c02h = -0x1.fffffffd9a965p-3;
        f64 const c03h = 0x1.99999929c014ap-3;
        f64 const c04h = -0x1.55554965c5ef8p-3;
        f64 const c05h = 0x1.249178e095642p-3;
        f64 const c06h = -0x1.ffecd2ace2b05p-4;
        f64 const c07h = 0x1.c67e15c92fa62p-4;
        f64 const c08h = -0x1.95efc62d0d9e2p-4;
        f64 const c09h = 0x1.64978be73f82p-4;
        f64 const c10h = -0x1.23a8be6d340bdp-4;
        f64 const c11h = 0x1.90c9dc821d8abp-5;
        f64 const c12h = -0x1.86f803157f71fp-6;
        f64 const c13h = 0x1.83f42484d9c35p-8;

        f64 suml = c13l;
        suml = __builtin_fma(suml, x, c12l);
        suml = __builtin_fma(suml, x, c11l);
        suml = __builtin_fma(suml, x, c10l);
        suml = __builtin_fma(suml, x, c00l);
        suml = __builtin_fma(suml, x, c09l);
        suml = __builtin_fma(suml, x, c08l);
        suml = __builtin_fma(suml, x, c07l);
        suml = __builtin_fma(suml, x, c06l);
        suml = __builtin_fma(suml, x, c05l);
        suml = __builtin_fma(suml, x, c04l);
        suml = __builtin_fma(suml, x, c03l);
        suml = __builtin_fma(suml, x, c02l);
        suml = __builtin_fma(suml, x, c01l);

        f64 sumh = c13h;
        sumh = __builtin_fma(sumh, x, c12h);
        sumh = __builtin_fma(sumh, x, c11h);
        sumh = __builtin_fma(sumh, x, c10h);
        sumh = __builtin_fma(sumh, x, c00h);
        sumh = __builtin_fma(sumh, x, c09h);
        sumh = __builtin_fma(sumh, x, c08h);
        sumh = __builtin_fma(sumh, x, c07h);
        sumh = __builtin_fma(sumh, x, c06h);
        sumh = __builtin_fma(sumh, x, c05h);
        sumh = __builtin_fma(sumh, x, c04h);
        sumh = __builtin_fma(sumh, x, c03h);
        sumh = __builtin_fma(sumh, x, c02h);
        sumh = __builtin_fma(sumh, x, c01h);

        f64 const break_point = 0x1.f0ed99bed9b2ep-8;

        f64 sum = x < break_point ? suml : sumh;

        return x + x * x * sum; 
    }
    else
    {
        volatile f64 u = 1.0 + x;
        volatile f64 d = u - 1.0;
    
        f64 safe_d = (d == 0.0) ? 1.0 : d; 
        return eagle_log(u) * (x / safe_d);
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
        /* Break point between the lower (code above) range polynomial and the upper (code below) range polynomial. */
        __m256d const break_point = _mm256_set1_pd(0x1.f0ed99bed9b2ep-8);
        __m256d const break_mask = _mm256_cmp_pd(x, break_point, _CMP_LE_OQ);

        __m256d const c00l = _mm256_set1_pd(-0x1p-1);
        __m256d const c01l = _mm256_set1_pd(0x1.5555555555165p-2);
        __m256d const c02l = _mm256_set1_pd(-0x1.00000000013a6p-2);
        __m256d const c03l = _mm256_set1_pd(0x1.999999b4d68bdp-3);
        __m256d const c04l = _mm256_set1_pd(-0x1.55554ce5cc951p-3);
        __m256d const c05l = _mm256_set1_pd(0x1.249375d994fa6p-3);
        __m256d const c06l = _mm256_set1_pd(-0x1.ffcf7eb545b5cp-4);
        __m256d const c07l = _mm256_set1_pd(0x1.c99c47194fe02p-4);
        __m256d const c08l = _mm256_set1_pd(-0x1.83879be1f1843p-4);
        __m256d const c09l = _mm256_set1_pd(0x1.f949fac831c28p-4);
        __m256d const c10l = _mm256_set1_pd(0x1.9562af4786612p-5);
        __m256d const c11l = _mm256_set1_pd(0x1.bdcd703e7e36dp-2);
        __m256d const c12l = _mm256_set1_pd(0x1.0a0fd969850b5p-1);
        __m256d const c13l = _mm256_set1_pd(0x1.1baa608cdc3abp-1);

        __m256d const c00h = _mm256_set1_pd(-0x1.fffffffffffcap-2);
        __m256d const c01h = _mm256_set1_pd(0x1.5555555552045p-2);
        __m256d const c02h = _mm256_set1_pd(-0x1.fffffffd9a965p-3);
        __m256d const c03h = _mm256_set1_pd(0x1.99999929c014ap-3);
        __m256d const c04h = _mm256_set1_pd(-0x1.55554965c5ef8p-3);
        __m256d const c05h = _mm256_set1_pd(0x1.249178e095642p-3);
        __m256d const c06h = _mm256_set1_pd(-0x1.ffecd2ace2b05p-4);
        __m256d const c07h = _mm256_set1_pd(0x1.c67e15c92fa62p-4);
        __m256d const c08h = _mm256_set1_pd(-0x1.95efc62d0d9e2p-4);
        __m256d const c09h = _mm256_set1_pd(0x1.64978be73f82p-4);
        __m256d const c10h = _mm256_set1_pd(-0x1.23a8be6d340bdp-4);
        __m256d const c11h = _mm256_set1_pd(0x1.90c9dc821d8abp-5);
        __m256d const c12h = _mm256_set1_pd(-0x1.86f803157f71fp-6);
        __m256d const c13h = _mm256_set1_pd(0x1.83f42484d9c35p-8);

        __m256d suml = c13l;
        suml = _mm256_fmadd_pd(suml, x, c12l);
        suml = _mm256_fmadd_pd(suml, x, c11l);
        suml = _mm256_fmadd_pd(suml, x, c10l);
        suml = _mm256_fmadd_pd(suml, x, c00l);
        suml = _mm256_fmadd_pd(suml, x, c09l);
        suml = _mm256_fmadd_pd(suml, x, c08l);
        suml = _mm256_fmadd_pd(suml, x, c07l);
        suml = _mm256_fmadd_pd(suml, x, c06l);
        suml = _mm256_fmadd_pd(suml, x, c05l);
        suml = _mm256_fmadd_pd(suml, x, c04l);
        suml = _mm256_fmadd_pd(suml, x, c03l);
        suml = _mm256_fmadd_pd(suml, x, c02l);
        suml = _mm256_fmadd_pd(suml, x, c01l);

        __m256d sumh = c13h;
        sumh = _mm256_fmadd_pd(sumh, x, c12h);
        sumh = _mm256_fmadd_pd(sumh, x, c11h);
        sumh = _mm256_fmadd_pd(sumh, x, c10h);
        sumh = _mm256_fmadd_pd(sumh, x, c00h);
        sumh = _mm256_fmadd_pd(sumh, x, c09h);
        sumh = _mm256_fmadd_pd(sumh, x, c08h);
        sumh = _mm256_fmadd_pd(sumh, x, c07h);
        sumh = _mm256_fmadd_pd(sumh, x, c06h);
        sumh = _mm256_fmadd_pd(sumh, x, c05h);
        sumh = _mm256_fmadd_pd(sumh, x, c04h);
        sumh = _mm256_fmadd_pd(sumh, x, c03h);
        sumh = _mm256_fmadd_pd(sumh, x, c02h);
        sumh = _mm256_fmadd_pd(sumh, x, c01h);

        __m256d sum = _mm256_blendv_pd(sumh, suml, break_mask);
        small_x_result = _mm256_add_pd(x, _mm256_mul_pd(_mm256_mul_pd(x, x), sum));
    }

    /* Handle NOT small x */
    __m256d not_small_x_result;
    {
        __m256d u = _mm256_add_pd(x, _mm256_set1_pd(1.0));
        __m256d d = _mm256_sub_pd(u, _mm256_set1_pd(1.0));
        __m256d safe_d = _mm256_blendv_pd(d, _mm256_set1_pd(1.0), _mm256_cmp_pd(d, _mm256_set1_pd(0.0), _CMP_EQ_OQ));
        not_small_x_result = _mm256_mul_pd(eagle_avx2_log_pd(u), _mm256_div_pd(x, safe_d));
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
        /* Break point between the lower (code above) range polynomial and the upper (code below) range polynomial. */
        __m512d const break_point = _mm512_set1_pd(0x1.f0ed99bed9b2ep-8);
        __mmask8 const break_mask = _mm512_cmp_pd_mask(x, break_point, _CMP_LE_OQ);

        __m512d const c00l = _mm512_set1_pd(-0x1p-1);
        __m512d const c01l = _mm512_set1_pd(0x1.5555555555165p-2);
        __m512d const c02l = _mm512_set1_pd(-0x1.00000000013a6p-2);
        __m512d const c03l = _mm512_set1_pd(0x1.999999b4d68bdp-3);
        __m512d const c04l = _mm512_set1_pd(-0x1.55554ce5cc951p-3);
        __m512d const c05l = _mm512_set1_pd(0x1.249375d994fa6p-3);
        __m512d const c06l = _mm512_set1_pd(-0x1.ffcf7eb545b5cp-4);
        __m512d const c07l = _mm512_set1_pd(0x1.c99c47194fe02p-4);
        __m512d const c08l = _mm512_set1_pd(-0x1.83879be1f1843p-4);
        __m512d const c09l = _mm512_set1_pd(0x1.f949fac831c28p-4);
        __m512d const c10l = _mm512_set1_pd(0x1.9562af4786612p-5);
        __m512d const c11l = _mm512_set1_pd(0x1.bdcd703e7e36dp-2);
        __m512d const c12l = _mm512_set1_pd(0x1.0a0fd969850b5p-1);
        __m512d const c13l = _mm512_set1_pd(0x1.1baa608cdc3abp-1);

        __m512d const c00h = _mm512_set1_pd(-0x1.fffffffffffcap-2);
        __m512d const c01h = _mm512_set1_pd(0x1.5555555552045p-2);
        __m512d const c02h = _mm512_set1_pd(-0x1.fffffffd9a965p-3);
        __m512d const c03h = _mm512_set1_pd(0x1.99999929c014ap-3);
        __m512d const c04h = _mm512_set1_pd(-0x1.55554965c5ef8p-3);
        __m512d const c05h = _mm512_set1_pd(0x1.249178e095642p-3);
        __m512d const c06h = _mm512_set1_pd(-0x1.ffecd2ace2b05p-4);
        __m512d const c07h = _mm512_set1_pd(0x1.c67e15c92fa62p-4);
        __m512d const c08h = _mm512_set1_pd(-0x1.95efc62d0d9e2p-4);
        __m512d const c09h = _mm512_set1_pd(0x1.64978be73f82p-4);
        __m512d const c10h = _mm512_set1_pd(-0x1.23a8be6d340bdp-4);
        __m512d const c11h = _mm512_set1_pd(0x1.90c9dc821d8abp-5);
        __m512d const c12h = _mm512_set1_pd(-0x1.86f803157f71fp-6);
        __m512d const c13h = _mm512_set1_pd(0x1.83f42484d9c35p-8);

        __m512d suml = c13l;
        suml = _mm512_fmadd_pd(suml, x, c12l);
        suml = _mm512_fmadd_pd(suml, x, c11l);
        suml = _mm512_fmadd_pd(suml, x, c10l);
        suml = _mm512_fmadd_pd(suml, x, c00l);
        suml = _mm512_fmadd_pd(suml, x, c09l);
        suml = _mm512_fmadd_pd(suml, x, c08l);
        suml = _mm512_fmadd_pd(suml, x, c07l);
        suml = _mm512_fmadd_pd(suml, x, c06l);
        suml = _mm512_fmadd_pd(suml, x, c05l);
        suml = _mm512_fmadd_pd(suml, x, c04l);
        suml = _mm512_fmadd_pd(suml, x, c03l);
        suml = _mm512_fmadd_pd(suml, x, c02l);
        suml = _mm512_fmadd_pd(suml, x, c01l);

        __m512d sumh = c13h;
        sumh = _mm512_fmadd_pd(sumh, x, c12h);
        sumh = _mm512_fmadd_pd(sumh, x, c11h);
        sumh = _mm512_fmadd_pd(sumh, x, c10h);
        sumh = _mm512_fmadd_pd(sumh, x, c00h);
        sumh = _mm512_fmadd_pd(sumh, x, c09h);
        sumh = _mm512_fmadd_pd(sumh, x, c08h);
        sumh = _mm512_fmadd_pd(sumh, x, c07h);
        sumh = _mm512_fmadd_pd(sumh, x, c06h);
        sumh = _mm512_fmadd_pd(sumh, x, c05h);
        sumh = _mm512_fmadd_pd(sumh, x, c04h);
        sumh = _mm512_fmadd_pd(sumh, x, c03h);
        sumh = _mm512_fmadd_pd(sumh, x, c02h);
        sumh = _mm512_fmadd_pd(sumh, x, c01h);

        __m512d sum = _mm512_mask_blend_pd(break_mask, sumh, suml);
        small_x_result = _mm512_add_pd(x, _mm512_mul_pd(_mm512_mul_pd(x, x), sum));
    }

    /* Handle NOT small x */
    __m512d not_small_x_result;
    {
        __m512d u = _mm512_add_pd(x, _mm512_set1_pd(1.0));
        __m512d d = _mm512_sub_pd(u, _mm512_set1_pd(1.0));
        __m512d safe_d = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(d, _mm512_set1_pd(0.0), _CMP_EQ_OQ), d, _mm512_set1_pd(1.0));
        not_small_x_result = _mm512_mul_pd(eagle_avx512_log_pd(u), _mm512_div_pd(x, safe_d));
    }

    __m512d final = _mm512_mask_blend_pd(small_mask, not_small_x_result, small_x_result);

    final = _mm512_mask_blend_pd(isnan_mask, final, _mm512_set1_pd(EAGLE_NAN));
    final = _mm512_mask_blend_pd(posinf_mask, final, _mm512_set1_pd(EAGLE_POS_INF));
    final = _mm512_mask_blend_pd(neg_one_mask, final, _mm512_set1_pd(EAGLE_NEG_INF));

    return final;
}

#endif 

