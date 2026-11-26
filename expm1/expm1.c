#include "../tests/eagle_test.h"

static inline f64
eagle_expm1(f64 x)
{
    /* High-precision constants for splitting LN(2) into HI and LO parts */
    f64 const INV_LN2 = +0x1.71547652b82fe0p+0000;
    f64 const LN2_HI  = +0x1.62e42fee000000p-0001;
    f64 const LN2_LO  = +0x1.a39ef35793c760p-0033;
    
    f64 const MAX_EXP   = +0x1.62e42fefa39efp+0009;  /* ~709.78  */
    f64 const MIN_EXP   = -0x1.2b708872320e1p+0005;  /* ~ -37.43 */

    /* Special value detection and early exits */
    if (eagle_isnan(x)) { return x; }
    if (x == EAGLE_POS_INF || x > MAX_EXP) { return EAGLE_POS_INF; }
    if (x == EAGLE_NEG_INF || x < MIN_EXP) { return -1.0; }

    /* Accurate range reduction: x = n * ln(2) + r */
    /* Using eagle_nearbyint() to match _MM_FROUND_TO_NEAREST_INT without exceptions */
    f64 n = eagle_nearbyint(x * INV_LN2);

    /* Calculate remainder carefully using scalar FMA: r = x - n*LN2_HI - n*LN2_LO */
    f64 r;
    _mm_store_sd(&r, _mm_fnmadd_sd(_mm_set_sd(n), _mm_set_sd(LN2_HI), _mm_set_sd(x)));
    _mm_store_sd(&r, _mm_fnmadd_sd(_mm_set_sd(n), _mm_set_sd(LN2_LO), _mm_set_sd(r)));

    f64 const c00 = 0x1p-1;
    f64 const c01 = 0x1.5555555555555p-3;
    f64 const c02 = 0x1.5555555555556p-5;
    f64 const c03 = 0x1.111111111137cp-7;
    f64 const c04 = 0x1.6c16c16c15fffp-10;
    f64 const c05 = 0x1.a01a019eaafc6p-13;
    f64 const c06 = 0x1.a01a01a815ae6p-16;
    f64 const c07 = 0x1.71de404b5e836p-19;
    f64 const c08 = 0x1.27e4d820b1d1cp-22;
    f64 const c09 = 0x1.ae4dbfe848851p-26;
    f64 const c10 = 0x1.1f78d10bcfc2dp-29;
    f64 const c11 = 0x1.81472fc1fcd7bp-33;

    f64 p = c11;
    p = __builtin_fma(p, r, c10);
    p = __builtin_fma(p, r, c09);
    p = __builtin_fma(p, r, c08);
    p = __builtin_fma(p, r, c07);
    p = __builtin_fma(p, r, c06);
    p = __builtin_fma(p, r, c05);
    p = __builtin_fma(p, r, c04);
    p = __builtin_fma(p, r, c03);
    p = __builtin_fma(p, r, c02);
    p = __builtin_fma(p, r, c01);
    p = __builtin_fma(p, r, c00);

    f64 p_full = r + r * r * p;

    /* Safe Scale Back: exp(x) = p * 2^n */
    /* To maintain bit-for-bit parity with the vector logic under extreme scales,
       we split the scale factor into two halves to completely safe-guard against 
       intermediate overflow or subnormal underflow. */
    i32 const n_int = (int)n;
    i32 const n1 = n_int / 2;
    i32 const n2 = n_int - n1;

    /* Build 2^n1 and 2^n2 directly via bit manipulation */
    u64 const biased_n1 = (u64)(n1 + 1023) << 52;
    u64 const biased_n2 = (u64)(n2 + 1023) << 52;
    u64 const biased_n = (u64)(n_int + 1023) << 52;
    
    f64 twon1, twon2, scale;
    __builtin_memcpy(&twon1, &biased_n1, sizeof(f64));
    __builtin_memcpy(&twon2, &biased_n2, sizeof(f64));
    __builtin_memcpy(&scale, &biased_n, sizeof(f64));

    if(n_int == 0)
    {
        /* Don't muck up the results by doing uneeded scaling. */
        return p_full;
    }
    else if(n_int > 53)
    {
        /* Prevent overflow at large values. */
        return twon2 * (twon1 + p_full * twon1);
    }
    else 
    {
        /* Sequentially apply scaling to safely approach subnormals or maximum limits */
        f64 term1 = __builtin_fma(scale, r, scale - 1.0);
        return __builtin_fma(scale, r * r * p, term1);
    }
}

#ifdef __AVX2__

static inline __m256d
eagle_avx2_expm1_pd(__m256d x)
{
    /* High-precision constants for splitting LN(2) into HI and LO parts */
    __m256d const INV_LN2 = _mm256_set1_pd(+0x1.71547652b82fe0p+0000);
    __m256d const LN2_HI  = _mm256_set1_pd(+0x1.62e42fee000000p-0001);
    __m256d const LN2_LO  = _mm256_set1_pd(+0x1.a39ef35793c760p-0033);
    
    __m256d const MAX_EXP   = _mm256_set1_pd(+0x1.62e42fefa39efp+0009);  /* ~709.78, expm1 of larger values overflows.    */
    __m256d const MIN_EXP   = _mm256_set1_pd(-0x1.2b708872320e1p+0005);  /* ~ -37.43, expm1 of smaller values underflows. */

    /* Special value detection */
    __m256d isnan_mask     = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);
    __m256d overflow_mask  = _mm256_cmp_pd(x, MAX_EXP, _CMP_GT_OQ);
    __m256d underflow_mask = _mm256_cmp_pd(x, MIN_EXP, _CMP_LT_OQ);

    __m256d result = x;  

    result = _mm256_blendv_pd(result, _mm256_set1_pd(EAGLE_POS_INF), overflow_mask);
    result = _mm256_blendv_pd(result, _mm256_set1_pd(-1.0), underflow_mask);

    __m256d all_specials = _mm256_or_pd(isnan_mask, _mm256_or_pd(overflow_mask, underflow_mask));
    
    /* Accurate range reduction: x = n * ln(2) + r */
    __m256d n = _mm256_round_pd(_mm256_mul_pd(x, INV_LN2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    /* Calculate remainder carefully: r = x - n*LN2_HI - n*LN2_LO */
    __m256d r = _mm256_fnmadd_pd(n, LN2_HI, x);
    r = _mm256_fnmadd_pd(n, LN2_LO, r);

    __m256d const c00 = _mm256_set1_pd(0x1p-1);
    __m256d const c01 = _mm256_set1_pd(0x1.5555555555555p-3);
    __m256d const c02 = _mm256_set1_pd(0x1.5555555555556p-5);
    __m256d const c03 = _mm256_set1_pd(0x1.111111111137cp-7);
    __m256d const c04 = _mm256_set1_pd(0x1.6c16c16c15fffp-10);
    __m256d const c05 = _mm256_set1_pd(0x1.a01a019eaafc6p-13);
    __m256d const c06 = _mm256_set1_pd(0x1.a01a01a815ae6p-16);
    __m256d const c07 = _mm256_set1_pd(0x1.71de404b5e836p-19);
    __m256d const c08 = _mm256_set1_pd(0x1.27e4d820b1d1cp-22);
    __m256d const c09 = _mm256_set1_pd(0x1.ae4dbfe848851p-26);
    __m256d const c10 = _mm256_set1_pd(0x1.1f78d10bcfc2dp-29);
    __m256d const c11 = _mm256_set1_pd(0x1.81472fc1fcd7bp-33);

    /* Standard Horner Evaluation */
    __m256d p = c11;
    p = _mm256_fmadd_pd(p, r, c10);
    p = _mm256_fmadd_pd(p, r, c09);
    p = _mm256_fmadd_pd(p, r, c08);
    p = _mm256_fmadd_pd(p, r, c07);
    p = _mm256_fmadd_pd(p, r, c06);
    p = _mm256_fmadd_pd(p, r, c05);
    p = _mm256_fmadd_pd(p, r, c04);
    p = _mm256_fmadd_pd(p, r, c03);
    p = _mm256_fmadd_pd(p, r, c02);
    p = _mm256_fmadd_pd(p, r, c01);
    p = _mm256_fmadd_pd(p, r, c00);

    __m256d p_full = _mm256_add_pd(r, _mm256_mul_pd(_mm256_mul_pd(r, r), p));

    /* AVX2-compatible arithmetic right shift by 1 for int64 */
    __m128i n_int32  = _mm256_cvtpd_epi32(n); 
    __m256i n_int64  = _mm256_cvtepi32_epi64(n_int32);
    __m256i n1 = _mm256_srli_epi64(n_int64, 1);
    __m256i high_bit = _mm256_and_si256(n_int64, _mm256_set1_epi64x(1ULL << 63));
    n1 = _mm256_or_si256(n1, high_bit);

    __m256i n2 = _mm256_sub_epi64(n_int64, n1);

    /* Build 2^n1 */
    __m256i biased_n1 = _mm256_add_epi64(n1, _mm256_set1_epi64x(1023));
    __m256d twon1     = _mm256_castsi256_pd(_mm256_slli_epi64(biased_n1, 52));

    /* Build 2^n2 */
    __m256i biased_n2 = _mm256_add_epi64(n2, _mm256_set1_epi64x(1023));
    __m256d twon2     = _mm256_castsi256_pd(_mm256_slli_epi64(biased_n2, 52));

    /* Build scale  */
    __m256i biased_n = _mm256_add_epi64(n_int64, _mm256_set1_epi64x(1023));
    __m256d scale    = _mm256_castsi256_pd(_mm256_slli_epi64(biased_n, 52));

    /* Just blend p_full in where n_int == 0 */
    __m256d zero_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(n_int64, _mm256_set1_epi64x(0)));
    result = _mm256_blendv_pd(result, p_full, zero_mask);

    /* Prevent overflow of large values and blend that in. */
    __m256d large_mask = _mm256_castsi256_pd(_mm256_cmpgt_epi64(n_int64, _mm256_set1_epi64x(53)));
    __m256d term = _mm256_mul_pd(twon2, _mm256_add_pd(twon1, _mm256_mul_pd(p_full, twon1)));
    result = _mm256_blendv_pd(result, term, large_mask);

    /* Apply scaling sequentially: normal_res = (p * 2^n1) * 2^n2 */
    __m256d remaining_mask = _mm256_or_pd(_mm256_or_pd(large_mask, zero_mask), all_specials);
    remaining_mask = _mm256_xor_pd(remaining_mask, _mm256_castsi256_pd(_mm256_set1_epi32(-1))); /* Bitwise NOT */

    __m256d term1 = _mm256_fmadd_pd(scale, r, _mm256_sub_pd(scale, _mm256_set1_pd(1.0)));
    term = _mm256_fmadd_pd(scale, _mm256_mul_pd(_mm256_mul_pd(r, r), p), term1);

    result = _mm256_blendv_pd(result, term, remaining_mask);

    return result;
}

#endif

#if ELK_AVX_512
static inline __m512d
eagle_avx512_expm1_pd(__m512d x)
{
    /* High-precision constants for splitting LN(2) into HI and LO parts */
    __m512d const INV_LN2 = _mm512_set1_pd(+0x1.71547652b82fe0p+0000);
    __m512d const LN2_HI  = _mm512_set1_pd(+0x1.62e42fee000000p-0001);
    __m512d const LN2_LO  = _mm512_set1_pd(+0x1.a39ef35793c760p-0033);
    
    __m512d const MAX_EXP   = _mm512_set1_pd(+0x1.62e42fefa39efp+0009);  /* ~709.78, expm1 of larger values overflows.   */
    __m512d const MIN_EXP   = _mm512_set1_pd(-0x1.2b708872320e1p+0005);  /* ~ -37.43, expm1 of smaller values underflows.*/

    /* Special value detection */
    __mmask8 isnan_mask     = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
    __mmask8 too_big_mask   = _mm512_cmp_pd_mask(x, MAX_EXP, _CMP_GT_OQ);
    __mmask8 too_small_mask = _mm512_cmp_pd_mask(x, MIN_EXP, _CMP_LT_OQ);

    __m512d result = x;  

    /* Handle specials with masks */
    result = _mm512_mask_mov_pd(result, too_big_mask, _mm512_set1_pd(EAGLE_POS_INF));
    result = _mm512_mask_mov_pd(result, too_small_mask, _mm512_set1_pd(-1.0));

    __mmask8 special_mask = _mm512_kor(isnan_mask, _mm512_kor(too_big_mask, too_small_mask));

    /* Accurate range reduction: x = n * ln(2) + r */
    __m512d n = _mm512_roundscale_pd(_mm512_mul_pd(x, INV_LN2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    /* Calculate remainder carefully: r = x - n*LN2_HI - n*LN2_LO */
    __m512d r = _mm512_fnmadd_pd(n, LN2_HI, x);
    r = _mm512_fnmadd_pd(n, LN2_LO, r);

    __m512d const c00 = _mm512_set1_pd(0x1p-1);
    __m512d const c01 = _mm512_set1_pd(0x1.5555555555555p-3);
    __m512d const c02 = _mm512_set1_pd(0x1.5555555555556p-5);
    __m512d const c03 = _mm512_set1_pd(0x1.111111111137cp-7);
    __m512d const c04 = _mm512_set1_pd(0x1.6c16c16c15fffp-10);
    __m512d const c05 = _mm512_set1_pd(0x1.a01a019eaafc6p-13);
    __m512d const c06 = _mm512_set1_pd(0x1.a01a01a815ae6p-16);
    __m512d const c07 = _mm512_set1_pd(0x1.71de404b5e836p-19);
    __m512d const c08 = _mm512_set1_pd(0x1.27e4d820b1d1cp-22);
    __m512d const c09 = _mm512_set1_pd(0x1.ae4dbfe848851p-26);
    __m512d const c10 = _mm512_set1_pd(0x1.1f78d10bcfc2dp-29);
    __m512d const c11 = _mm512_set1_pd(0x1.81472fc1fcd7bp-33);

    /* Standard Horner Evaluation */
    __m512d p = c11;
    p = _mm512_fmadd_pd(p, r, c10);
    p = _mm512_fmadd_pd(p, r, c09);
    p = _mm512_fmadd_pd(p, r, c08);
    p = _mm512_fmadd_pd(p, r, c07);
    p = _mm512_fmadd_pd(p, r, c06);
    p = _mm512_fmadd_pd(p, r, c05);
    p = _mm512_fmadd_pd(p, r, c04);
    p = _mm512_fmadd_pd(p, r, c03);
    p = _mm512_fmadd_pd(p, r, c02);
    p = _mm512_fmadd_pd(p, r, c01);
    p = _mm512_fmadd_pd(p, r, c00);

    __m512d p_full = _mm512_add_pd(r, _mm512_mul_pd(_mm512_mul_pd(r, r), p));

    /* Safe Scale Back: exp(x) = p * 2^n */
    /* To maintain bit-for-bit parity with the vector logic under extreme scales,
       we split the scale factor into two halves to completely safe-guard against 
       intermediate overflow or subnormal underflow. */
    __m512i n_int = _mm512_cvtpd_epi64(n);
    __m512i n1 = _mm512_srli_epi64(n_int, 1);
    __m512i n2 = _mm512_sub_epi64(n_int, n1);

    __m512d twon1 = _mm512_scalef_pd(_mm512_set1_pd(1.0), _mm512_cvtepi64_pd(n1));
    __m512d twon2 = _mm512_scalef_pd(_mm512_set1_pd(1.0), _mm512_cvtepi64_pd(n2));
    __m512d scale = _mm512_scalef_pd(_mm512_set1_pd(1.0), _mm512_cvtepi64_pd(n_int));

    /* Don't muck up the results by doing uneeded scaling. */
    __mmask8 zero_mask = _mm512_cmpeq_epi64_mask(n_int, _mm512_set1_epi64(0));
    result = _mm512_mask_mov_pd(result, zero_mask, p_full);
    
    /* Prevent overflow at large values. */
    __mmask8 large_mask = _mm512_cmpgt_epi64_mask(n_int, _mm512_set1_epi64(53));
    __m512d term = _mm512_mul_pd(twon2, _mm512_add_pd(twon1, _mm512_mul_pd(p_full, twon1)));
    result = _mm512_mask_mov_pd(result, large_mask, term);
    
    /* Apply scaling sequentially: normal_res = (p * 2^n1) * 2^n2 */
    __mmask8 remaining_mask = ~(special_mask | large_mask | zero_mask);
    __m512d term1 = _mm512_fmadd_pd(scale, r, _mm512_sub_pd(scale, _mm512_set1_pd(1.0)));
    term = _mm512_fmadd_pd(scale, _mm512_mul_pd(_mm512_mul_pd(r, r), p), term1);
    result = _mm512_mask_mov_pd(result, remaining_mask, term);

    return result;
}

#endif 

