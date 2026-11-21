#include "../tests/eagle_test.h"

static inline f64
eagle_log(f64 x)
{
#define SQRT2   +0x1.6a09e667f3bcdp+0000

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
    i32 adj = (s >= SQRT2);
    exponent += adj;
    s *= adj ? 0.5 : 1.0;

    f64 const z = s - 1.0; 
    f64 const f = z / (2.0 + z);
    f64 const f2 = f * f;

    f64 const Lg1 = 0x1.5555555555594p-1;
    f64 const Lg2 = 0x1.999999997f733p-2;
    f64 const Lg3 = 0x1.249249423bc69p-2;
    f64 const Lg4 = 0x1.c71c51d5e1a8fp-3;
    f64 const Lg5 = 0x1.746648289dbfcp-3;
    f64 const Lg6 = 0x1.39a0e50a4daf7p-3;
    f64 const Lg7 = 0x1.2f0d0a2a8996cp-3;

    f64 R = Lg7;
    R = __builtin_fma(R, f2, Lg6);
    R = __builtin_fma(R, f2, Lg5);
    R = __builtin_fma(R, f2, Lg4);
    R = __builtin_fma(R, f2, Lg3);
    R = __builtin_fma(R, f2, Lg2);
    R = __builtin_fma(R, f2, Lg1);
    R = __builtin_fma(R, f2, 0.0);

    f64 const hfsq = 0.5 * z * z;

    f64 const sum = z - (hfsq - f * (hfsq + R));

    f64 const LN2_HI = 0x1.62e42fefa39efp-1;   /* 53 bits   */
    f64 const LN2_LO = 0x1.abc9e3b39803fp-56;  /* remainder */

    f64 const tmp = __builtin_fma((f64)exponent, LN2_HI, sum);
    f64 const err = __builtin_fma((f64)exponent, LN2_HI, -tmp) + sum;   /* error from first addition */
    f64 const result = __builtin_fma((f64)exponent, LN2_LO, tmp + err);

    return result;

#undef SQRT2
}

#ifdef __AVX2__

static inline __m256d
eagle_convert_epi64_to_pd_avx2(__m256i x)
{
    /* Extract lane 0, 2, 4, 6 (the low 32-bit ints of each 64-bit element) */
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
    __m128i i32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(x, idx));
    return _mm256_cvtepi32_pd(i32);
}

static inline __m256d
eagle_negate_avx2(__m256d v)
{
    /* Creates a vector of [ -0.0, -0.0, -0.0, -0.0 ] (0x8000000000000000ULL per lane) */
    __m256d const sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x8000000000000000ULL));
    return _mm256_xor_pd(v, sign_mask);
}

static inline __m256d
eagle_avx2_log_pd(__m256d x)
{
#define SQRT2      _mm256_set1_pd(+0x1.6a09e667f3bcdp+0000)
#define LN2_HI     _mm256_set1_pd(+0x1.62e42fefa39efp-0001)
#define LN2_LO     _mm256_set1_pd(+0x1.abc9e3b39803fp-0056)
#define SUBNORM_S  _mm256_set1_pd(0x1.0p54)
#define SUBNORM_E  54LL

    __m256d isnan_mask = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);
    __m256d const lt_zero_mask = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_LT_OQ);
    isnan_mask = _mm256_or_pd(isnan_mask, lt_zero_mask);

    __m256d const posinf_mask = _mm256_cmp_pd(x, _mm256_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);
    __m256d const zero_mask   = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_EQ_OQ);

    /* Identify strictly positive subnormals: 0.0 < x < 2^-1022 */
    __m256d const gt_zero_mask = _mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_GT_OQ);
    __m256d const lt_min_normal = _mm256_cmp_pd(x, _mm256_set1_pd(0x1.0p-1022), _CMP_LT_OQ);
    __m256d const subnormal_mask = _mm256_and_pd(gt_zero_mask, lt_min_normal);

    /* Normalize subnormal lanes by 2^54 so they become normal numbers in [2^-1020, 2^-968) */
    __m256d const x_scaled = _mm256_mul_pd(x, SUBNORM_S);
    __m256d const x_norm = _mm256_blendv_pd(x, x_scaled, subnormal_mask);

    __m256i const pun = _mm256_castpd_si256(x_norm);

    /* Decompose into a number in the range [1.0, 2.0) and the exponent. */
    __m256i const mantissa = _mm256_and_si256(pun, _mm256_set1_epi64x((long long)0x000FFFFFFFFFFFFFLL));
    __m256i const raw_exponent = _mm256_and_si256(_mm256_srli_epi64(pun, 52), _mm256_set1_epi64x(0x7FFLL));
    __m256i exponent = _mm256_sub_epi64(raw_exponent, _mm256_set1_epi64x(1023LL));

    /* Subtract 54 from exponent on subnormal lanes to compensate for multiplying x by 2^54 */
    __m256i const exp_adj = _mm256_sub_epi64(exponent, _mm256_set1_epi64x(SUBNORM_E));
    exponent = _mm256_blendv_epi8(exponent, exp_adj, _mm256_castpd_si256(subnormal_mask));

    __m256i const significand_pun = _mm256_or_si256(_mm256_slli_epi64(_mm256_set1_epi64x(1023ULL), 52), mantissa);
    __m256d s = _mm256_castsi256_pd(significand_pun);

    /* Map from [1.0, 2.0) to [sqrt(2)/2, sqrt(2)) */
    __m256d gte_sqrt2_mask = _mm256_cmp_pd(s, SQRT2, _CMP_GE_OQ);

    __m256d const sa = _mm256_mul_pd(s, _mm256_set1_pd(0.5));
    __m256i const one = _mm256_set1_epi64x(1LL);
    __m256i const ea = _mm256_add_epi64(exponent, one);

    s = _mm256_blendv_pd(s, sa, gte_sqrt2_mask);
    exponent = _mm256_blendv_epi8(exponent, ea, _mm256_castpd_si256(gte_sqrt2_mask));

    __m256d const z = _mm256_sub_pd(s, _mm256_set1_pd(1.0));
    __m256d const f = _mm256_div_pd(z, _mm256_add_pd(z, _mm256_set1_pd(2.0)));
    __m256d const f2 = _mm256_mul_pd(f, f);

    __m256d const Lg1 = _mm256_set1_pd(0x1.5555555555594p-1);
    __m256d const Lg2 = _mm256_set1_pd(0x1.999999997f733p-2);
    __m256d const Lg3 = _mm256_set1_pd(0x1.249249423bc69p-2);
    __m256d const Lg4 = _mm256_set1_pd(0x1.c71c51d5e1a8fp-3);
    __m256d const Lg5 = _mm256_set1_pd(0x1.746648289dbfcp-3);
    __m256d const Lg6 = _mm256_set1_pd(0x1.39a0e50a4daf7p-3);
    __m256d const Lg7 = _mm256_set1_pd(0x1.2f0d0a2a8996cp-3);

    __m256d R = Lg7;
    R = _mm256_fmadd_pd(R, f2, Lg6);
    R = _mm256_fmadd_pd(R, f2, Lg5);
    R = _mm256_fmadd_pd(R, f2, Lg4);
    R = _mm256_fmadd_pd(R, f2, Lg3);
    R = _mm256_fmadd_pd(R, f2, Lg2);
    R = _mm256_fmadd_pd(R, f2, Lg1);
    R = _mm256_mul_pd(R, f2);

    __m256d const hfsq = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(z, z));
    __m256d const sum = _mm256_sub_pd(z, _mm256_sub_pd(hfsq, _mm256_mul_pd(f, _mm256_add_pd(hfsq, R))));

    __m256d const exponent_pd = eagle_convert_epi64_to_pd_avx2(exponent);
    __m256d const tmp = _mm256_fmadd_pd(exponent_pd, LN2_HI, sum);
    __m256d const err = _mm256_add_pd(_mm256_fmadd_pd(exponent_pd, LN2_HI, eagle_negate_avx2(tmp)), sum);
    __m256d result = _mm256_fmadd_pd(exponent_pd, LN2_LO, _mm256_add_pd(tmp, err));

    result = _mm256_blendv_pd(result, _mm256_set1_pd(EAGLE_NAN), isnan_mask);
    result = _mm256_blendv_pd(result, _mm256_set1_pd(EAGLE_POS_INF), posinf_mask);
    result = _mm256_blendv_pd(result, _mm256_set1_pd(EAGLE_NEG_INF), zero_mask);

    return result;

#undef SUBNORM_E
#undef SUBNORM_S
#undef SQRT2
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

static inline __m512d eagle_avx512_negate_pd(__m512d v)
{
    __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000ULL);
    __m512i vi = _mm512_castpd_si512(v);
    return _mm512_castsi512_pd(_mm512_xor_si512(vi, sign_mask));
}

static inline __m512d
eagle_avx512_log_pd(__m512d x)
{
    __m512d const ZERO      = _mm512_set1_pd(0.0);
    __m512d const ONE       = _mm512_set1_pd(1.0);
    __m512d const HALF      = _mm512_set1_pd(0.5);
    __m512d const TWO       = _mm512_set1_pd(2.0);
    __m512d const SQRT2     = _mm512_set1_pd(0x1.6a09e667f3bcdp+0);
    __m512d const LN2_HI    = _mm512_set1_pd(+0x1.62e42fefa39efp-0001);
    __m512d const LN2_LO    = _mm512_set1_pd(+0x1.abc9e3b39803fp-0056);

    /* Special cases */
    __mmask8 isnan_mask  = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
    __mmask8 neg_mask    = _mm512_cmp_pd_mask(x, ZERO, _CMP_LT_OQ);
    __mmask8 posinf_mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(EAGLE_POS_INF), _CMP_EQ_OQ);
    __mmask8 zero_mask   = _mm512_cmp_pd_mask(x, ZERO, _CMP_EQ_OQ);

    __mmask8 special_mask = isnan_mask | neg_mask | posinf_mask | zero_mask;

    /* AVX512-native decomposition */
    __m512d s = _mm512_getmant_pd(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    __m512i exponent = _mm512_cvtpd_epi64(_mm512_getexp_pd(x));

    /* Range reduction: map [1.0, 2.0) → [√2/2, √2) */
    __mmask8 ge_sqrt2_mask = _mm512_cmp_pd_mask(s, SQRT2,   _CMP_GE_OQ);

    s = _mm512_mask_mul_pd(s, ge_sqrt2_mask, s, HALF);

    __m512i exp_adj = exponent;
    exp_adj = _mm512_mask_add_epi64(exp_adj, ge_sqrt2_mask, exp_adj, _mm512_set1_epi64(1));

    __m512d const z = _mm512_sub_pd(s, ONE); 
    __m512d const f = _mm512_div_pd(z, _mm512_add_pd(z, TWO));
    __m512d const f2 = _mm512_mul_pd(f, f);

    __m512d const Lg1 = _mm512_set1_pd(0x1.5555555555594p-1);
    __m512d const Lg2 = _mm512_set1_pd(0x1.999999997f733p-2);
    __m512d const Lg3 = _mm512_set1_pd(0x1.249249423bc69p-2);
    __m512d const Lg4 = _mm512_set1_pd(0x1.c71c51d5e1a8fp-3);
    __m512d const Lg5 = _mm512_set1_pd(0x1.746648289dbfcp-3);
    __m512d const Lg6 = _mm512_set1_pd(0x1.39a0e50a4daf7p-3);
    __m512d const Lg7 = _mm512_set1_pd(0x1.2f0d0a2a8996cp-3);

    __m512d R = Lg7;
    R = _mm512_fmadd_pd(R, f2, Lg6);
    R = _mm512_fmadd_pd(R, f2, Lg5);
    R = _mm512_fmadd_pd(R, f2, Lg4);
    R = _mm512_fmadd_pd(R, f2, Lg3);
    R = _mm512_fmadd_pd(R, f2, Lg2);
    R = _mm512_fmadd_pd(R, f2, Lg1);
    R = _mm512_mul_pd(R, f2);

    __m512d const hfsq = _mm512_mul_pd(HALF, _mm512_mul_pd(z, z));
    __m512d const sum = _mm512_sub_pd(z, _mm512_sub_pd(hfsq, _mm512_mul_pd(f, _mm512_add_pd(hfsq, R))));

    /* exponent * ln(2) with accurate summation */
    __m512d const exp_pd = _mm512_cvtepi64_pd(exp_adj);
    __m512d const tmp = _mm512_fmadd_pd(exp_pd, LN2_HI, sum);
    __m512d const err = _mm512_add_pd(_mm512_fmadd_pd(exp_pd, LN2_HI, eagle_avx512_negate_pd(tmp)), sum);
    __m512d result = _mm512_fmadd_pd(exp_pd, LN2_LO, _mm512_add_pd(tmp, err));

    /* Apply special cases */
    result = _mm512_mask_blend_pd(special_mask, result, _mm512_set1_pd(EAGLE_NAN));
    result = _mm512_mask_blend_pd(posinf_mask, result, _mm512_set1_pd(EAGLE_POS_INF));
    result = _mm512_mask_blend_pd(zero_mask, result, _mm512_set1_pd(EAGLE_NEG_INF));

    return result;
}

#endif 

