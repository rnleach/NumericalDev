#include "../tests/eagle_test.h"

static inline f64
eagle_log(f64 x)
{
#define SQRT2   +0x1.6a09e667f3bcdp+0000
#define SQRT2_2 +0x1.6a09e667f3bcdp-0001

    if(eagle_isnan(x) | (x < 0.0)) { return EAGLE_NAN; }
    if(x == 0.0) { return EAGLE_NEG_INF; }
    if(x == EAGLE_POS_INF) { return EAGLE_POS_INF; }

    union 
    {
        f64 x;
        u64 bits;
    } pun, significand;

    pun.x = x;

    /* Decompose into a number in the range [1.0, 2.0) and the exponent.                                            */
    u64 mantissa = pun.bits & 0x000FFFFFFFFFFFFFLL; /* Mask out the mantissa bits, remember missing leading 1.      */
    u64 raw_exponent = (pun.bits >> 52) & 0x7FF;    /* extract the 11 exponent bits.                                */
    i32 exponent = (i32)(raw_exponent - 1023);      /* Subtract the exponent bias.                                  */
    significand.bits = (1023ULL << 52) | mantissa;  /* Add bias back in for exponent, but force exponent to be 0.0. */

    /* Map from [1.0, 2.0) to [sqrt(2)/2, sqrt(2))                                                                  */
    f64 s = significand.x;
    i32 adj = (s >= SQRT2) - (s < SQRT2_2);  /* +1, 0, or -1 */
    exponent += adj;
    s *= (adj == 0) ? 1.0 : (adj > 0 ? 0.5 : 2.0);

    f64 const z = s - 1.0; 

    f64 const c00l = 0;
    f64 const c01l = 0x1.fffffffffff2p-1;
    f64 const c02l = -0x1.0000000035318p-1;
    f64 const c03l = 0x1.55555512a818ep-2;
    f64 const c04l = -0x1.00001050b6c75p-2;
    f64 const c05l = 0x1.99957db02b90ap-3;
    f64 const c06l = -0x1.55a3169b319ddp-3;
    f64 const c07l = 0x1.20f424c1c16a1p-3;
    f64 const c08l = -0x1.1c12fa8498de7p-3;
    f64 const c09l = 0x1.52319fa3479e1p-5;
    f64 const c10l = -0x1.506bd6add6276p-2;
    f64 const c11l = -0x1.6eb563dcf0e6ep-2;
    f64 const c12l = -0x1.0b82a6c4bf6f5p-1;

    f64 const c00h = 0;
    f64 const c01h = 0x1.ffffffffffefcp-1;
    f64 const c02h = -0x1.ffffffffa66d2p-2;
    f64 const c03h = 0x1.5555552c5edacp-2;
    f64 const c04h = -0x1.fffff144f2d0ap-3;
    f64 const c05h = 0x1.99983969b5ff6p-3;
    f64 const c06h = -0x1.5541c7d06ee1dp-3;
    f64 const c07h = 0x1.23e02d2a040d4p-3;
    f64 const c08h = -0x1.f783cfbaba881p-4;
    f64 const c09h = 0x1.a3a33fa2edacdp-4;
    f64 const c10h = -0x1.32d08b3702861p-4;
    f64 const c11h = 0x1.49f5e66d51e9fp-5;
    f64 const c12h = -0x1.7032a2334f732p-7;

    f64 suml = c12l;
    suml = __builtin_fma(suml, z, c11l);
    suml = __builtin_fma(suml, z, c10l);
    suml = __builtin_fma(suml, z, c09l);
    suml = __builtin_fma(suml, z, c08l);
    suml = __builtin_fma(suml, z, c07l);
    suml = __builtin_fma(suml, z, c06l);
    suml = __builtin_fma(suml, z, c05l);
    suml = __builtin_fma(suml, z, c04l);
    suml = __builtin_fma(suml, z, c03l);
    suml = __builtin_fma(suml, z, c02l);
    suml = __builtin_fma(suml, z, c01l);
    suml = __builtin_fma(suml, z, c00l);

    f64 sumh = c12h;
    sumh = __builtin_fma(sumh, z, c11h);
    sumh = __builtin_fma(sumh, z, c10h);
    sumh = __builtin_fma(sumh, z, c09h);
    sumh = __builtin_fma(sumh, z, c08h);
    sumh = __builtin_fma(sumh, z, c07h);
    sumh = __builtin_fma(sumh, z, c06h);
    sumh = __builtin_fma(sumh, z, c05h);
    sumh = __builtin_fma(sumh, z, c04h);
    sumh = __builtin_fma(sumh, z, c03h);
    sumh = __builtin_fma(sumh, z, c02h);
    sumh = __builtin_fma(sumh, z, c01h);
    sumh = __builtin_fma(sumh, z, c00h);

    f64 const sum = z < 0.0 ? suml : sumh;

    f64 const LN2_HI = +0x1.62e42fee000000p-0001;
    f64 const LN2_LO = +0x1.a39ef35793c760p-0033;

    f64 const exp_term_hi = (f64)exponent * LN2_HI;
    f64 const exp_term_lo = (f64)exponent * LN2_LO;

    f64 const tmp = exp_term_hi + sum;
    f64 const err = (exp_term_hi - tmp) + sum;   /* error from first addition */
    f64 const result = tmp + (err + exp_term_lo);

    return result;

#undef SQRT2
#undef SQRT2_2
}

#ifdef __AVX2__

static inline __m256d
eagle_convert_epi64_to_pd_avx2(__m256i x)
{
    /* Convert signed 64-bit integers to f64. Works because |exponent| is small. */
    const __m256d bias = _mm256_set1_pd(0x1.0p52);           /* 2^52 */
    const __m256i magic = _mm256_set1_epi64x(0x4330000000000000LL); /* 2^52 as biased f64 */

    __m256i biased = _mm256_add_epi64(x, magic);
    __m256d as_dbl = _mm256_castsi256_pd(biased);
    return _mm256_sub_pd(as_dbl, bias);
}

static inline __m256d
eagle_avx2_log_pd(__m256d x)
{
#define SQRT2   _mm256_set1_pd(+0x1.6a09e667f3bcdp+0000)
#define SQRT2_2 _mm256_set1_pd(+0x1.6a09e667f3bcdp-0001)
#define LN2_HI  _mm256_set1_pd(+0x1.62e42fee000000p-0001)
#define LN2_LO  _mm256_set1_pd(+0x1.a39ef35793c760p-0033)

    __m256d isnan_mask   = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);
    __m256d const lt_zero_mask = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_LT_OQ);
    isnan_mask = _mm256_or_pd(isnan_mask, lt_zero_mask);

    __m256d const posinf_mask  = _mm256_cmp_pd(x, _mm256_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);
    __m256d const zero_mask    = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_EQ_OQ);

    __m256i pun = _mm256_castpd_si256(x);

    /* Decompose into a number in the range [1.0, 2.0) and the exponent. */
    __m256i const mantissa = _mm256_and_si256(pun, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL));
    __m256i const raw_exponent = _mm256_and_si256(_mm256_srli_epi64(pun, 52), _mm256_set1_epi64x(0x7FF));
    __m256i exponent = _mm256_sub_epi64(raw_exponent, _mm256_set1_epi64x(1023));
    __m256i const significand_pun = _mm256_or_si256(_mm256_slli_epi64(_mm256_set1_epi64x(1023ULL), 52), mantissa);
    __m256d significand = _mm256_castsi256_pd(significand_pun);

    /* Map from [1.0, 2.0) to [sqrt(2)/2, sqrt(2)) */
    __m256d gte_sqrt2_mask = _mm256_cmp_pd(significand, SQRT2, _CMP_GE_OQ);
    __m256d lt_sqrt2_mask = _mm256_cmp_pd(significand, SQRT2_2, _CMP_LT_OQ);

    __m256d sa = _mm256_mul_pd(significand, _mm256_set1_pd(0.5));
    __m256d sb = _mm256_mul_pd(significand, _mm256_set1_pd(2.0));
    __m256i one = _mm256_set1_epi64x(1LL);
    __m256i ea = _mm256_add_epi64(exponent, one);
    __m256i eb = _mm256_sub_epi64(exponent, one);

    significand = _mm256_blendv_pd(significand, sa, gte_sqrt2_mask);
    significand = _mm256_blendv_pd(significand, sb, lt_sqrt2_mask);
    significand = _mm256_sub_pd(significand, _mm256_set1_pd(1.0));
    exponent = _mm256_blendv_epi8(exponent, ea, _mm256_castpd_si256(gte_sqrt2_mask));
    exponent = _mm256_blendv_epi8(exponent, eb, _mm256_castpd_si256(lt_sqrt2_mask));

    __m256d low_mask = _mm256_cmp_pd(significand, _mm256_set1_pd(0.0), _CMP_LT_OQ);

    /* Polynomial approximation to log(1 + r) for r < 0 */
    __m256d const c00l = _mm256_set1_pd(0);
    __m256d const c01l = _mm256_set1_pd(0x1.fffffffffff2p-1);
    __m256d const c02l = _mm256_set1_pd(-0x1.0000000035318p-1);
    __m256d const c03l = _mm256_set1_pd(0x1.55555512a818ep-2);
    __m256d const c04l = _mm256_set1_pd(-0x1.00001050b6c75p-2);
    __m256d const c05l = _mm256_set1_pd(0x1.99957db02b90ap-3);
    __m256d const c06l = _mm256_set1_pd(-0x1.55a3169b319ddp-3);
    __m256d const c07l = _mm256_set1_pd(0x1.20f424c1c16a1p-3);
    __m256d const c08l = _mm256_set1_pd(-0x1.1c12fa8498de7p-3);
    __m256d const c09l = _mm256_set1_pd(0x1.52319fa3479e1p-5);
    __m256d const c10l = _mm256_set1_pd(-0x1.506bd6add6276p-2);
    __m256d const c11l = _mm256_set1_pd(-0x1.6eb563dcf0e6ep-2);
    __m256d const c12l = _mm256_set1_pd(-0x1.0b82a6c4bf6f5p-1);

    __m256d const c00h = _mm256_set1_pd(0);
    __m256d const c01h = _mm256_set1_pd(0x1.ffffffffffefcp-1);
    __m256d const c02h = _mm256_set1_pd(-0x1.ffffffffa66d2p-2);
    __m256d const c03h = _mm256_set1_pd(0x1.5555552c5edacp-2);
    __m256d const c04h = _mm256_set1_pd(-0x1.fffff144f2d0ap-3);
    __m256d const c05h = _mm256_set1_pd(0x1.99983969b5ff6p-3);
    __m256d const c06h = _mm256_set1_pd(-0x1.5541c7d06ee1dp-3);
    __m256d const c07h = _mm256_set1_pd(0x1.23e02d2a040d4p-3);
    __m256d const c08h = _mm256_set1_pd(-0x1.f783cfbaba881p-4);
    __m256d const c09h = _mm256_set1_pd(0x1.a3a33fa2edacdp-4);
    __m256d const c10h = _mm256_set1_pd(-0x1.32d08b3702861p-4);
    __m256d const c11h = _mm256_set1_pd(0x1.49f5e66d51e9fp-5);
    __m256d const c12h = _mm256_set1_pd(-0x1.7032a2334f732p-7);

    __m256d suml = c12l;
    suml = _mm256_fmadd_pd(suml, significand, c11l);
    suml = _mm256_fmadd_pd(suml, significand, c10l);
    suml = _mm256_fmadd_pd(suml, significand, c09l);
    suml = _mm256_fmadd_pd(suml, significand, c08l);
    suml = _mm256_fmadd_pd(suml, significand, c07l);
    suml = _mm256_fmadd_pd(suml, significand, c06l);
    suml = _mm256_fmadd_pd(suml, significand, c05l);
    suml = _mm256_fmadd_pd(suml, significand, c04l);
    suml = _mm256_fmadd_pd(suml, significand, c03l);
    suml = _mm256_fmadd_pd(suml, significand, c02l);
    suml = _mm256_fmadd_pd(suml, significand, c01l);
    suml = _mm256_fmadd_pd(suml, significand, c00l);

    __m256d sumh = c12h;
    sumh = _mm256_fmadd_pd(sumh, significand, c11h);
    sumh = _mm256_fmadd_pd(sumh, significand, c10h);
    sumh = _mm256_fmadd_pd(sumh, significand, c09h);
    sumh = _mm256_fmadd_pd(sumh, significand, c08h);
    sumh = _mm256_fmadd_pd(sumh, significand, c07h);
    sumh = _mm256_fmadd_pd(sumh, significand, c06h);
    sumh = _mm256_fmadd_pd(sumh, significand, c05h);
    sumh = _mm256_fmadd_pd(sumh, significand, c04h);
    sumh = _mm256_fmadd_pd(sumh, significand, c03h);
    sumh = _mm256_fmadd_pd(sumh, significand, c02h);
    sumh = _mm256_fmadd_pd(sumh, significand, c01h);
    sumh = _mm256_fmadd_pd(sumh, significand, c00h);

    __m256d const sum = _mm256_blendv_pd(sumh, suml, low_mask);

    __m256d const exponent_pd = eagle_convert_epi64_to_pd_avx2(exponent); 
    __m256d const exp_term_hi = _mm256_mul_pd(exponent_pd, LN2_HI);
    __m256d const exp_term_lo = _mm256_mul_pd(exponent_pd, LN2_LO);

    __m256d const tmp = _mm256_add_pd(exp_term_hi, sum);
    __m256d const err = _mm256_add_pd(_mm256_sub_pd(exp_term_hi, tmp), sum);
    __m256d final = _mm256_add_pd(tmp, _mm256_add_pd(err, exp_term_lo));

    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_NAN), isnan_mask);
    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_POS_INF), posinf_mask);
    final = _mm256_blendv_pd(final, _mm256_set1_pd(EAGLE_NEG_INF), zero_mask);

    return final;

#undef SQRT2
#undef SQRT2_2
#undef LN2_HI
#undef LN2_LO
}

#endif

#if ELK_AVX_512

static inline __m512d
eagle_convert_epi64_to_pd_avx512(__m512i x)
{
    /* Convert signed 64-bit integers to f64. Works because |exponent| is small. */
    const __m512d bias = _mm512_set1_pd(0x1.0p52);           /* 2^52 */
    const __m512i magic = _mm512_set1_epi64(0x4330000000000000LL); /* 2^52 as biased f64 */

    __m512i biased = _mm512_add_epi64(x, magic);
    __m512d as_dbl = _mm512_castsi512_pd(biased);
    return _mm512_sub_pd(as_dbl, bias);
}

static inline __m512d
eagle_avx512_log_pd(__m512d x)
{
    const __m512d ZERO      = _mm512_set1_pd(0.0);
    const __m512d ONE       = _mm512_set1_pd(1.0);
    const __m512d HALF      = _mm512_set1_pd(0.5);
    const __m512d TWO       = _mm512_set1_pd(2.0);
    const __m512d SQRT2     = _mm512_set1_pd(0x1.6a09e667f3bcdp+0);
    const __m512d SQRT2_2   = _mm512_set1_pd(0x1.6a09e667f3bcdp-1);
    const __m512d LN2_HI    = _mm512_set1_pd(+0x1.62e42fee000000p-0001);
    const __m512d LN2_LO    = _mm512_set1_pd(+0x1.a39ef35793c760p-0033);

    /* Special cases */
    __mmask8 isnan_mask  = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
    __mmask8 neg_mask    = _mm512_cmp_pd_mask(x, ZERO, _CMP_LT_OQ);
    __mmask8 posinf_mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);
    __mmask8 zero_mask   = _mm512_cmp_pd_mask(x, ZERO, _CMP_EQ_OQ);

    __mmask8 special_mask = isnan_mask | neg_mask | posinf_mask | zero_mask;

    /* === AVX512-native decomposition === */
    __m512d significand = _mm512_getmant_pd(x,
                                            _MM_MANT_NORM_1_2,     // [1.0, 2.0)
                                            _MM_MANT_SIGN_src);    // preserve sign

    __m512i exponent = _mm512_cvtpd_epi64(_mm512_getexp_pd(x));

    /* Range reduction: map [1.0, 2.0) → [√2/2, √2) */
    __mmask8 ge_sqrt2_mask = _mm512_cmp_pd_mask(significand, SQRT2,   _CMP_GE_OQ);
    __mmask8 lt_s2_2_mask  = _mm512_cmp_pd_mask(significand, SQRT2_2, _CMP_LT_OQ);

    __m512d r = significand;

    r = _mm512_mask_mul_pd(r, ge_sqrt2_mask, r, HALF);
    r = _mm512_mask_mul_pd(r, lt_s2_2_mask,  r, TWO);

    __m512i exp_adj = exponent;
    exp_adj = _mm512_mask_add_epi64(exp_adj, ge_sqrt2_mask, exp_adj, _mm512_set1_epi64(1));
    exp_adj = _mm512_mask_sub_epi64(exp_adj, lt_s2_2_mask,  exp_adj, _mm512_set1_epi64(1));

    r = _mm512_sub_pd(r, ONE); 

    __mmask8 low_mask = _mm512_cmp_pd_mask(r, _mm512_set1_pd(0.0), _CMP_LT_OQ);

    /* Polynomial approximation to log(1 + r) for r < 0 */
    __m512d const c00l = _mm512_set1_pd(0);
    __m512d const c01l = _mm512_set1_pd(0x1.fffffffffff2p-1);
    __m512d const c02l = _mm512_set1_pd(-0x1.0000000035318p-1);
    __m512d const c03l = _mm512_set1_pd(0x1.55555512a818ep-2);
    __m512d const c04l = _mm512_set1_pd(-0x1.00001050b6c75p-2);
    __m512d const c05l = _mm512_set1_pd(0x1.99957db02b90ap-3);
    __m512d const c06l = _mm512_set1_pd(-0x1.55a3169b319ddp-3);
    __m512d const c07l = _mm512_set1_pd(0x1.20f424c1c16a1p-3);
    __m512d const c08l = _mm512_set1_pd(-0x1.1c12fa8498de7p-3);
    __m512d const c09l = _mm512_set1_pd(0x1.52319fa3479e1p-5);
    __m512d const c10l = _mm512_set1_pd(-0x1.506bd6add6276p-2);
    __m512d const c11l = _mm512_set1_pd(-0x1.6eb563dcf0e6ep-2);
    __m512d const c12l = _mm512_set1_pd(-0x1.0b82a6c4bf6f5p-1);

    __m512d const c00h = _mm512_set1_pd(0);
    __m512d const c01h = _mm512_set1_pd(0x1.ffffffffffefcp-1);
    __m512d const c02h = _mm512_set1_pd(-0x1.ffffffffa66d2p-2);
    __m512d const c03h = _mm512_set1_pd(0x1.5555552c5edacp-2);
    __m512d const c04h = _mm512_set1_pd(-0x1.fffff144f2d0ap-3);
    __m512d const c05h = _mm512_set1_pd(0x1.99983969b5ff6p-3);
    __m512d const c06h = _mm512_set1_pd(-0x1.5541c7d06ee1dp-3);
    __m512d const c07h = _mm512_set1_pd(0x1.23e02d2a040d4p-3);
    __m512d const c08h = _mm512_set1_pd(-0x1.f783cfbaba881p-4);
    __m512d const c09h = _mm512_set1_pd(0x1.a3a33fa2edacdp-4);
    __m512d const c10h = _mm512_set1_pd(-0x1.32d08b3702861p-4);
    __m512d const c11h = _mm512_set1_pd(0x1.49f5e66d51e9fp-5);
    __m512d const c12h = _mm512_set1_pd(-0x1.7032a2334f732p-7);

    __m512d suml = c12l;
    suml = _mm512_fmadd_pd(suml, r, c11l);
    suml = _mm512_fmadd_pd(suml, r, c10l);
    suml = _mm512_fmadd_pd(suml, r, c09l);
    suml = _mm512_fmadd_pd(suml, r, c08l);
    suml = _mm512_fmadd_pd(suml, r, c07l);
    suml = _mm512_fmadd_pd(suml, r, c06l);
    suml = _mm512_fmadd_pd(suml, r, c05l);
    suml = _mm512_fmadd_pd(suml, r, c04l);
    suml = _mm512_fmadd_pd(suml, r, c03l);
    suml = _mm512_fmadd_pd(suml, r, c02l);
    suml = _mm512_fmadd_pd(suml, r, c01l);
    suml = _mm512_fmadd_pd(suml, r, c00l);

    __m512d sumh = c12h;
    sumh = _mm512_fmadd_pd(sumh, r, c11h);
    sumh = _mm512_fmadd_pd(sumh, r, c10h);
    sumh = _mm512_fmadd_pd(sumh, r, c09h);
    sumh = _mm512_fmadd_pd(sumh, r, c08h);
    sumh = _mm512_fmadd_pd(sumh, r, c07h);
    sumh = _mm512_fmadd_pd(sumh, r, c06h);
    sumh = _mm512_fmadd_pd(sumh, r, c05h);
    sumh = _mm512_fmadd_pd(sumh, r, c04h);
    sumh = _mm512_fmadd_pd(sumh, r, c03h);
    sumh = _mm512_fmadd_pd(sumh, r, c02h);
    sumh = _mm512_fmadd_pd(sumh, r, c01h);
    sumh = _mm512_fmadd_pd(sumh, r, c00h);

    __m512d const p = _mm512_mask_mov_pd(sumh, low_mask, suml);

    /* exponent * ln(2) with accurate summation */
    __m512d exp_pd = _mm512_cvtepi64_pd(exp_adj);
    __m512d exp_hi = _mm512_mul_pd(exp_pd, LN2_HI);
    __m512d exp_lo = _mm512_mul_pd(exp_pd, LN2_LO);

    __m512d tmp = _mm512_add_pd(exp_hi, p);
    __m512d err = _mm512_add_pd(_mm512_sub_pd(exp_hi, tmp), p);

    __m512d result = _mm512_add_pd(tmp, _mm512_add_pd(err, exp_lo));

    /* Apply special cases */
    result = _mm512_mask_blend_pd(special_mask, result, _mm512_set1_pd(EAGLE_NAN));
    result = _mm512_mask_blend_pd(posinf_mask, result, _mm512_set1_pd(EAGLE_POS_INF));
    result = _mm512_mask_blend_pd(zero_mask, result, _mm512_set1_pd(EAGLE_NEG_INF));

    return result;
}

#endif 

