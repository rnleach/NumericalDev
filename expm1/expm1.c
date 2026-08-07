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

    f64 const c09l = 0x1.74e6047471e31p-26;
    f64 const c08l = 0x1.2552d8b0e8d0dp-22;
    f64 const c07l = 0x1.71bd43a1344f8p-19;
    f64 const c06l = 0x1.a019016b7cdf3p-16;
    f64 const c05l = 0x1.a019fcd0c2641p-13;
    f64 const c04l = 0x1.6c16c15e1545dp-10;
    f64 const c03l = 0x1.11111110f9ea5p-7;
    f64 const c02l = 0x1.55555555552fap-5;
    f64 const c01l = 0x1.5555555555554p-3;
    f64 const c00l = 0x1p-1;

    f64 const c09h = 0x1.f11722efc30b3p-26;
    f64 const c08h = 0x1.24c5bbba4267p-22;
    f64 const c07h = 0x1.72070db75a794p-19;
    f64 const c06h = 0x1.a018c180c98d3p-16;
    f64 const c05h = 0x1.a01a07a764185p-13;
    f64 const c04h = 0x1.6c16c15a8a8e3p-10;
    f64 const c03h = 0x1.111111112e015p-7;
    f64 const c02h = 0x1.5555555555265p-5;
    f64 const c01h = 0x1.5555555555557p-3;
    f64 const c00h = 0x1p-1;

    f64 pl = c09l;
    pl = __builtin_fma(pl, r, c08l);
    pl = __builtin_fma(pl, r, c07l);
    pl = __builtin_fma(pl, r, c06l);
    pl = __builtin_fma(pl, r, c05l);
    pl = __builtin_fma(pl, r, c04l);
    pl = __builtin_fma(pl, r, c03l);
    pl = __builtin_fma(pl, r, c02l);
    pl = __builtin_fma(pl, r, c01l);
    pl = __builtin_fma(pl, r, c00l);

    f64 ph = c09h;
    ph = __builtin_fma(ph, r, c08h);
    ph = __builtin_fma(ph, r, c07h);
    ph = __builtin_fma(ph, r, c06h);
    ph = __builtin_fma(ph, r, c05h);
    ph = __builtin_fma(ph, r, c04h);
    ph = __builtin_fma(ph, r, c03h);
    ph = __builtin_fma(ph, r, c02h);
    ph = __builtin_fma(ph, r, c01h);
    ph = __builtin_fma(ph, r, c00h);

    f64 p_tail = r < -0x1.c642ccb7dbaccp-90 ? pl : ph;
    f64 p_full = r + r * r * p_tail;

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
        return __builtin_fma(scale, r * r * p_tail, term1);
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

    /* Range mask - polynomial is broken into two ranges, a low and a high range */
    __m256d low_mask = _mm256_cmp_pd(r, _mm256_set1_pd(-0x1.c642ccb7dbaccp-90), _CMP_LT_OQ);

    /* Low range polynomial coefficients */
    __m256d const c9l = _mm256_set1_pd(0x1.74e6047471e31p-26);
    __m256d const c8l = _mm256_set1_pd(0x1.2552d8b0e8d0dp-22);
    __m256d const c7l = _mm256_set1_pd(0x1.71bd43a1344f8p-19);
    __m256d const c6l = _mm256_set1_pd(0x1.a019016b7cdf3p-16);
    __m256d const c5l = _mm256_set1_pd(0x1.a019fcd0c2641p-13);
    __m256d const c4l = _mm256_set1_pd(0x1.6c16c15e1545dp-10);
    __m256d const c3l = _mm256_set1_pd(0x1.11111110f9ea5p-7);
    __m256d const c2l = _mm256_set1_pd(0x1.55555555552fap-5);
    __m256d const c1l = _mm256_set1_pd(0x1.5555555555554p-3);
    __m256d const c0l = _mm256_set1_pd(0x1p-1);

    /* High range polynomial coefficients */
    __m256d const c9h = _mm256_set1_pd(0x1.f11722efc30b3p-26);
    __m256d const c8h = _mm256_set1_pd(0x1.24c5bbba4267p-22);
    __m256d const c7h = _mm256_set1_pd(0x1.72070db75a794p-19);
    __m256d const c6h = _mm256_set1_pd(0x1.a018c180c98d3p-16);
    __m256d const c5h = _mm256_set1_pd(0x1.a01a07a764185p-13);
    __m256d const c4h = _mm256_set1_pd(0x1.6c16c15a8a8e3p-10);
    __m256d const c3h = _mm256_set1_pd(0x1.111111112e015p-7);
    __m256d const c2h = _mm256_set1_pd(0x1.5555555555265p-5);
    __m256d const c1h = _mm256_set1_pd(0x1.5555555555557p-3);
    __m256d const c0h = _mm256_set1_pd(0x1p-1);

    /* Standard Horner Evaluation */
    __m256d pl = c9l;
    pl = _mm256_fmadd_pd(pl, r, c8l);
    pl = _mm256_fmadd_pd(pl, r, c7l);
    pl = _mm256_fmadd_pd(pl, r, c6l);
    pl = _mm256_fmadd_pd(pl, r, c5l);
    pl = _mm256_fmadd_pd(pl, r, c4l);
    pl = _mm256_fmadd_pd(pl, r, c3l);
    pl = _mm256_fmadd_pd(pl, r, c2l);
    pl = _mm256_fmadd_pd(pl, r, c1l);
    pl = _mm256_fmadd_pd(pl, r, c0l);

    __m256d ph = c9h;
    ph = _mm256_fmadd_pd(ph, r, c8h);
    ph = _mm256_fmadd_pd(ph, r, c7h);
    ph = _mm256_fmadd_pd(ph, r, c6h);
    ph = _mm256_fmadd_pd(ph, r, c5h);
    ph = _mm256_fmadd_pd(ph, r, c4h);
    ph = _mm256_fmadd_pd(ph, r, c3h);
    ph = _mm256_fmadd_pd(ph, r, c2h);
    ph = _mm256_fmadd_pd(ph, r, c1h);
    ph = _mm256_fmadd_pd(ph, r, c0h);

    __m256d p_tail = _mm256_blendv_pd(ph, pl, low_mask);
    __m256d p_full = _mm256_add_pd(r, _mm256_mul_pd(_mm256_mul_pd(r, r), p_tail));

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
    term = _mm256_fmadd_pd(scale, _mm256_mul_pd(_mm256_mul_pd(r, r), p_tail), term1);

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

    /* Range mask - polynomial is broken into two ranges, a low and a high range */
    __mmask8 low_mask = _mm512_cmp_pd_mask(r, _mm512_set1_pd(-0x1.c642ccb7dbaccp-90), _CMP_LT_OQ);

    /* Low range polynomial coefficients */
    __m512d const c9l = _mm512_set1_pd(0x1.74e6047471e31p-26);
    __m512d const c8l = _mm512_set1_pd(0x1.2552d8b0e8d0dp-22);
    __m512d const c7l = _mm512_set1_pd(0x1.71bd43a1344f8p-19);
    __m512d const c6l = _mm512_set1_pd(0x1.a019016b7cdf3p-16);
    __m512d const c5l = _mm512_set1_pd(0x1.a019fcd0c2641p-13);
    __m512d const c4l = _mm512_set1_pd(0x1.6c16c15e1545dp-10);
    __m512d const c3l = _mm512_set1_pd(0x1.11111110f9ea5p-7);
    __m512d const c2l = _mm512_set1_pd(0x1.55555555552fap-5);
    __m512d const c1l = _mm512_set1_pd(0x1.5555555555554p-3);
    __m512d const c0l = _mm512_set1_pd(0x1p-1);

    /* High range polynomial coefficients */
    __m512d const c9h = _mm512_set1_pd(0x1.f11722efc30b3p-26);
    __m512d const c8h = _mm512_set1_pd(0x1.24c5bbba4267p-22);
    __m512d const c7h = _mm512_set1_pd(0x1.72070db75a794p-19);
    __m512d const c6h = _mm512_set1_pd(0x1.a018c180c98d3p-16);
    __m512d const c5h = _mm512_set1_pd(0x1.a01a07a764185p-13);
    __m512d const c4h = _mm512_set1_pd(0x1.6c16c15a8a8e3p-10);
    __m512d const c3h = _mm512_set1_pd(0x1.111111112e015p-7);
    __m512d const c2h = _mm512_set1_pd(0x1.5555555555265p-5);
    __m512d const c1h = _mm512_set1_pd(0x1.5555555555557p-3);
    __m512d const c0h = _mm512_set1_pd(0x1p-1);

    /* Standard Horner Evaluation */
    __m512d pl = c9l;
    pl = _mm512_fmadd_pd(pl, r, c8l);
    pl = _mm512_fmadd_pd(pl, r, c7l);
    pl = _mm512_fmadd_pd(pl, r, c6l);
    pl = _mm512_fmadd_pd(pl, r, c5l);
    pl = _mm512_fmadd_pd(pl, r, c4l);
    pl = _mm512_fmadd_pd(pl, r, c3l);
    pl = _mm512_fmadd_pd(pl, r, c2l);
    pl = _mm512_fmadd_pd(pl, r, c1l);
    pl = _mm512_fmadd_pd(pl, r, c0l);

    __m512d ph = c9h;
    ph = _mm512_fmadd_pd(ph, r, c8h);
    ph = _mm512_fmadd_pd(ph, r, c7h);
    ph = _mm512_fmadd_pd(ph, r, c6h);
    ph = _mm512_fmadd_pd(ph, r, c5h);
    ph = _mm512_fmadd_pd(ph, r, c4h);
    ph = _mm512_fmadd_pd(ph, r, c3h);
    ph = _mm512_fmadd_pd(ph, r, c2h);
    ph = _mm512_fmadd_pd(ph, r, c1h);
    ph = _mm512_fmadd_pd(ph, r, c0h);

    __m512d p_tail = _mm512_mask_mov_pd(ph, low_mask, pl);
    __m512d p_full = _mm512_add_pd(r, _mm512_mul_pd(_mm512_mul_pd(r, r), p_tail));

    /* Safe Scale Back: exp(x) = p * 2^n */
    /* To maintain bit-for-bit parity with the vector logic under extreme scales,
       we split the scale factor into two halves to completely safe-guard against 
       intermediate overflow or subnormal underflow. */
    __m512i n_int = _mm512_cvtpd_epi64(n);
    __m512i n1 = _mm512_srli_epi64(n_int, 1);
    __m512i n2 = _mm512_sub_epi64(n_int, n1);

    /* Build 2^n1 and 2^n2 directly via bit manipulation */
    __m512i const biased_n1 = _mm512_slli_epi64(_mm512_add_epi64(n1, _mm512_set1_epi64(1023)), 52);
    __m512i const biased_n2 = _mm512_slli_epi64(_mm512_add_epi64(n2, _mm512_set1_epi64(1023)), 52);
    __m512i const biased_n = _mm512_slli_epi64(_mm512_add_epi64(n_int, _mm512_set1_epi64(1023)), 52);

    __m512d twon1 = _mm512_castsi512_pd(biased_n1);
    __m512d twon2 = _mm512_castsi512_pd(biased_n2);
    __m512d scale = _mm512_castsi512_pd(biased_n);

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
    term = _mm512_fmadd_pd(scale, _mm512_mul_pd(_mm512_mul_pd(r, r), p_tail), term1);
    result = _mm512_mask_mov_pd(result, remaining_mask, term);

    return result;
}

#endif 

