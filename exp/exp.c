#include "../tests/eagle_test.h"

static inline f64
eagle_exp(f64 x)
{
    /* High-precision constants for splitting LN(2) into HI and LO parts */
    f64 const INV_LN2 = +0x1.71547652b82fe0p+0000;
    f64 const LN2_HI  = +0x1.62e42fefa39ef0p-0001;
    f64 const LN2_LO  = +0x1.abc9e3b39803f8p-0056;
    
    f64 const MAX_EXP   = +0x1.62e42fefa39efp+0009;  /* ~709.78 */
    f64 const MIN_EXP   = -0x1.74910d52d3050p+0009;  /* ~745.13 */

    /* Special value detection and early exits */
    if (eagle_isnan(x)) { return x; }
    if (x == EAGLE_POS_INF || x > MAX_EXP) { return EAGLE_POS_INF; }
    if (x == EAGLE_NEG_INF || x < MIN_EXP) { return 0.0; }

    /* Accurate range reduction: x = n * ln(2) + r */
    /* Using eagle_nearbyint() to match _MM_FROUND_TO_NEAREST_INT without exceptions */
    f64 n = eagle_nearbyint(x * INV_LN2);

    /* Calculate remainder carefully using scalar FMA: r = x - n*LN2_HI - n*LN2_LO */
    f64 r;
    _mm_store_sd(&r, _mm_fnmadd_sd(_mm_set_sd(n), _mm_set_sd(LN2_HI), _mm_set_sd(x)));
    _mm_store_sd(&r, _mm_fnmadd_sd(_mm_set_sd(n), _mm_set_sd(LN2_LO), _mm_set_sd(r)));

    /* Exact 15-term Minimax Horner scheme evaluation for exp(r) */
    f64 const c15 = -0x1.d945b463d6c94p-34;
    f64 const c14 = +0x1.bbadbc0979032p-37;
    f64 const c13 = +0x1.ce87fa264b15cp-33;
    f64 const c12 = +0x1.1edd947c37df1p-29;
    f64 const c11 = +0x1.ae3c31db580fdp-26;
    f64 const c10 = +0x1.27e500668efb2p-22;
    f64 const c09 = +0x1.71de41ce844b6p-19;
    f64 const c08 = +0x1.a01a019ea5942p-16;
    f64 const c07 = +0x1.a01a019ea160fp-13;
    f64 const c06 = +0x1.6c16c16c16f53p-10;
    f64 const c05 = +0x1.111111111135ap-7;
    f64 const c04 = +0x1.5555555555555p-5;
    f64 const c03 = +0x1.5555555555555p-3;
    f64 const c02 = +0x1p-1;
    //f64 const c01 = +0x1p0;

    //f64 const p0 = __builtin_fma(c01, r, 1.0);
    f64 const p0 = r + 1.0;
    f64 const p1 = __builtin_fma(c03, r, c02);
    f64 const p2 = __builtin_fma(c05, r, c04);
    f64 const p3 = __builtin_fma(c07, r, c06);
    f64 const p4 = __builtin_fma(c09, r, c08);
    f64 const p5 = __builtin_fma(c11, r, c10);
    f64 const p6 = __builtin_fma(c13, r, c12);
    f64 const p7 = __builtin_fma(c15, r, c14);

    f64 const r2 = r * r;
    f64 p = p7;
    p = __builtin_fma(p, r2, p6);
    p = __builtin_fma(p, r2, p5);
    p = __builtin_fma(p, r2, p4);
    p = __builtin_fma(p, r2, p3);
    p = __builtin_fma(p, r2, p2);
    p = __builtin_fma(p, r2, p1);
    p = __builtin_fma(p, r2, p0);

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
    
    f64 twon1, twon2;
    __builtin_memcpy(&twon1, &biased_n1, sizeof(f64));
    __builtin_memcpy(&twon2, &biased_n2, sizeof(f64));

    /* Sequentially apply scaling to safely approach subnormals or maximum limits */
    return (p * twon1) * twon2;
}

#ifdef __AVX2__

static inline __m256d
eagle_avx2_exp_pd(__m256d x)
{
    /* High-precision constants for splitting LN(2) into HI and LO parts */
    __m256d const INV_LN2 = _mm256_set1_pd(+0x1.71547652b82fe0p+0000);
    __m256d const LN2_HI  = _mm256_set1_pd(+0x1.62e42fefa39ef0p-0001);
    __m256d const LN2_LO  = _mm256_set1_pd(+0x1.abc9e3b39803f8p-0056);
    
    __m256d const MAX_EXP   = _mm256_set1_pd(+0x1.62e42fefa39efp+0009);  /* ~709.78 */
    __m256d const MIN_EXP   = _mm256_set1_pd(-0x1.74910d52d3050p+0009);  /* ~-745.13 */

    /* Special value detection */
    __m256d isnan_mask     = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);
    __m256d overflow_mask  = _mm256_cmp_pd(x, MAX_EXP, _CMP_GT_OQ);
    __m256d underflow_mask = _mm256_cmp_pd(x, MIN_EXP, _CMP_LT_OQ);

    __m256d result = x;  

    result = _mm256_blendv_pd(result, _mm256_set1_pd(EAGLE_POS_INF), overflow_mask);
    result = _mm256_blendv_pd(result, _mm256_set1_pd(0.0), underflow_mask);

    __m256d all_specials = _mm256_or_pd(isnan_mask, _mm256_or_pd(overflow_mask, underflow_mask));
    
    /* Accurate range reduction: x = n * ln(2) + r */
    __m256d n = _mm256_round_pd(_mm256_mul_pd(x, INV_LN2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    /* Calculate remainder carefully: r = x - n*LN2_HI - n*LN2_LO */
    __m256d r = _mm256_fnmadd_pd(n, LN2_HI, x);
    r = _mm256_fnmadd_pd(n, LN2_LO, r);

    /* Exact 14-term Minimax Horner scheme evaluation for exp(r) */
    __m256d const c14 = _mm256_set1_pd(+0x1.7c736aa2ba40ap-0036);
    __m256d const c13 = _mm256_set1_pd(+0x1.6e74d2b2921a7p-0033);
    __m256d const c12 = _mm256_set1_pd(+0x1.1e62043af923dp-0029);
    __m256d const c11 = _mm256_set1_pd(+0x1.ae5a3e68ca5d7p-0026);
    __m256d const c10 = _mm256_set1_pd(+0x1.27e52520dc37cp-0022);
    __m256d const c9  = _mm256_set1_pd(+0x1.71de3d518ab16p-0019);
    __m256d const c8  = _mm256_set1_pd(+0x1.a01a01945b6cbp-0016);
    __m256d const c7  = _mm256_set1_pd(+0x1.a01a019f43752p-0013);
    __m256d const c6  = _mm256_set1_pd(+0x1.6c16c16c184b0p-0010);
    __m256d const c5  = _mm256_set1_pd(+0x1.11111111112d2p-0007);
    __m256d const c4  = _mm256_set1_pd(+0x1.5555555555553p-0005);
    __m256d const c3  = _mm256_set1_pd(+0x1.5555555555555p-0003);
    __m256d const c2  = _mm256_set1_pd(+0x1.0000000000000p-0001);
    __m256d const c1  = _mm256_set1_pd(+0x1.0000000000000p+0000);

    /* Standard Horner Evaluation */
    __m256d p = c14;
    p = _mm256_fmadd_pd(p, r, c13);
    p = _mm256_fmadd_pd(p, r, c12);
    p = _mm256_fmadd_pd(p, r, c11);
    p = _mm256_fmadd_pd(p, r, c10);
    p = _mm256_fmadd_pd(p, r, c9);
    p = _mm256_fmadd_pd(p, r, c8);
    p = _mm256_fmadd_pd(p, r, c7);
    p = _mm256_fmadd_pd(p, r, c6);
    p = _mm256_fmadd_pd(p, r, c5);
    p = _mm256_fmadd_pd(p, r, c4);
    p = _mm256_fmadd_pd(p, r, c3);
    p = _mm256_fmadd_pd(p, r, c2);
    p = _mm256_fmadd_pd(p, r, c1);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));

    /* AVX2-compatible arithmetic right shift by 1 for int64 */
    __m128i n_int32  = _mm256_cvtpd_epi32(n); 
    __m256i n_int64  = _mm256_cvtepi32_epi64(n_int32);
    __m256i n1 = _mm256_srli_epi64(n_int64, 1);
    __m256i high_bit = _mm256_and_si256(n_int64, _mm256_set1_epi64x(1ULL << 63));
    n1 = _mm256_or_si256(n1, high_bit);

    __m256i n2 = _mm256_sub_epi64(n_int64, n1);

    /* Build 2^n1 */
    __m256i biased_n1   = _mm256_add_epi64(n1, _mm256_set1_epi64x(1023));
    __m256d twon1       = _mm256_castsi256_pd(_mm256_slli_epi64(biased_n1, 52));

    /* Build 2^n2 */
    __m256i biased_n2   = _mm256_add_epi64(n2, _mm256_set1_epi64x(1023));
    __m256d twon2       = _mm256_castsi256_pd(_mm256_slli_epi64(biased_n2, 52));

    /* Apply scaling sequentially: normal_res = (p * 2^n1) * 2^n2 */
    __m256d normal_res = _mm256_mul_pd(_mm256_mul_pd(p, twon1), twon2);

    /* Blend normal and special results */
    result = _mm256_blendv_pd(normal_res, result, all_specials);

    return result;
}

#endif

#if ELK_AVX_512
static inline __m512d
eagle_avx512_exp_pd(__m512d x)
{
    /* High-precision constants for splitting LN(2) into HI and LO parts */
    __m512d const INV_LN2 = _mm512_set1_pd(+0x1.71547652b82fep+0000);
    __m512d const LN2_HI  = _mm512_set1_pd(+0x1.62e42fefa39efp-1);
    __m512d const LN2_LO  = _mm512_set1_pd(+0x1.abc9e3b39803f8p-56);
    
    __m512d const MAX_EXP   = _mm512_set1_pd(+0x1.62e42fefa39efp+0009);  /* ~709.78, exp of larger values overflows.    */
    __m512d const MIN_EXP   = _mm512_set1_pd(-0x1.74910d52d3050p+0009);  /* ~-745.13, exp of smaller values underflows. */

    /* Special value detection */
    __mmask8 isnan_mask     = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
    __mmask8 posinf_mask    = _mm512_cmp_pd_mask(x, _mm512_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);
    __mmask8 neginf_mask    = _mm512_cmp_pd_mask(x, _mm512_set1_pd(EAGLE_NEG_INF), _CMP_EQ_OQ);
    __mmask8 too_big_mask   = _mm512_cmp_pd_mask(x, MAX_EXP, _CMP_GT_OQ);
    __mmask8 too_small_mask = _mm512_cmp_pd_mask(x, MIN_EXP, _CMP_LT_OQ);

    __m512d result = x;  

    /* Handle specials with masks */
    result = _mm512_mask_mov_pd(result, too_big_mask | posinf_mask, _mm512_set1_pd(EAGLE_POS_INF));
    result = _mm512_mask_mov_pd(result, too_small_mask | neginf_mask, _mm512_set1_pd(0.0));

    __mmask8 normal_mask = _mm512_knot(_mm512_kor(isnan_mask, _mm512_kor(posinf_mask, _mm512_kor(neginf_mask,
                                      _mm512_kor(too_big_mask, too_small_mask)))));

    if (_mm512_kortestz(normal_mask, normal_mask)) { return result; } /* All lanes handled by special cases. */

    /* Accurate range reduction: x = n * ln(2) + r */
    __m512d n = _mm512_roundscale_pd(_mm512_mul_pd(x, INV_LN2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    /* Calculate remainder carefully: r = x - n*LN2_HI - n*LN2_LO */
    __m512d r = _mm512_fnmadd_pd(n, LN2_HI, x);
    r = _mm512_fnmadd_pd(n, LN2_LO, r);

    /* Exact 14-term Minimax Horner scheme evaluation for exp(r) */
    __m512d const c14 = _mm512_set1_pd(+0x1.7c736aa2ba40ap-0036);
    __m512d const c13 = _mm512_set1_pd(+0x1.6e74d2b2921a7p-0033);
    __m512d const c12 = _mm512_set1_pd(+0x1.1e62043af923dp-0029);
    __m512d const c11 = _mm512_set1_pd(+0x1.ae5a3e68ca5d7p-0026);
    __m512d const c10 = _mm512_set1_pd(+0x1.27e52520dc37cp-0022);
    __m512d const c9  = _mm512_set1_pd(+0x1.71de3d518ab16p-0019);
    __m512d const c8  = _mm512_set1_pd(+0x1.a01a01945b6cbp-0016);
    __m512d const c7  = _mm512_set1_pd(+0x1.a01a019f43752p-0013);
    __m512d const c6  = _mm512_set1_pd(+0x1.6c16c16c184b0p-0010);
    __m512d const c5  = _mm512_set1_pd(+0x1.11111111112d2p-0007);
    __m512d const c4  = _mm512_set1_pd(+0x1.5555555555553p-0005);
    __m512d const c3  = _mm512_set1_pd(+0x1.5555555555555p-0003);
    __m512d const c2  = _mm512_set1_pd(+0x1.0000000000000p-0001);
    __m512d const c1  = _mm512_set1_pd(+0x1.0000000000000p+0000);

    /* Standard Horner Evaluation: p = (((c14 * r + c13) * r + c12) ... ) * r + 1.0 */
    __m512d p = c14;
    p = _mm512_fmadd_pd(p, r, c13);
    p = _mm512_fmadd_pd(p, r, c12);
    p = _mm512_fmadd_pd(p, r, c11);
    p = _mm512_fmadd_pd(p, r, c10);
    p = _mm512_fmadd_pd(p, r, c9);
    p = _mm512_fmadd_pd(p, r, c8);
    p = _mm512_fmadd_pd(p, r, c7);
    p = _mm512_fmadd_pd(p, r, c6);
    p = _mm512_fmadd_pd(p, r, c5);
    p = _mm512_fmadd_pd(p, r, c4);
    p = _mm512_fmadd_pd(p, r, c3);
    p = _mm512_fmadd_pd(p, r, c2);
    p = _mm512_fmadd_pd(p, r, c1);
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0));

    /* Scale back: exp(x) = p * 2^n */
    __m512d normal_res = _mm512_scalef_pd(p, n);

    /* Blend normal and special results */
    result = _mm512_mask_mov_pd(result, normal_mask, normal_res);

    return result;
}

#endif 

