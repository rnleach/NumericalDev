#ifndef EAGLE_MATH_H_
#define EAGLE_MATH_H_

#include <stdint.h>
#include <stddef.h>

#include <immintrin.h>
#include "elk.h"

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                                          Math
 *
 *-------------------------------------------------------------------------------------------------------------------------*/

#define EAGLE_POS_INF __builtin_inf()
#define EAGLE_NEG_INF (-__builtin_inf())
#define EAGLE_NAN     __builtin_nan("")

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                                        Scalar
 *
 *-------------------------------------------------------------------------------------------------------------------------*/
#define eagle_isnan(x) eagle_isnan_bits(x)
static inline f64 eagle_nextafter(f64 x, f64 y);
static inline i64 f64_ulp_distance(f64 a, f64 b, int64_t max_threshold);
static inline b32 eagle_isnan_bits(f64 x);
static inline f64 eagle_nearbyint(f64 x);
static inline f64 eagle_exp(f64 x);
static inline f64 eagle_log(f64 x);

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                                         AVX2
 *
 *-------------------------------------------------------------------------------------------------------------------------*/
#ifdef __AVX2__

static inline __m256d eagle_avx2_exp_pd(__m256d x);
static inline __m256d eagle_avx2_log_pd(__m256d x);

#endif

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                                        AVX512
 *
 *-------------------------------------------------------------------------------------------------------------------------*/
#if ELK_AVX_512 

static inline __m512d eagle_avx512_exp_pd(__m512d x);
static inline __m512d eagle_avx512_log_pd(__m512d x);

#endif

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                                    Implementations
 *
 *-------------------------------------------------------------------------------------------------------------------------*/

static inline f64
eagle_nextafter(f64 x, f64 y) 
{
    /* Handle NaNs */
    if (x != x || y != y) return x + y; 
    if (x == y) return y;

    /* View double bits as a 64-bit unsigned integer */
    union { f64 d; u64 u; } ux;
    ux.d = x;

    if (x == 0.0)
    {
        /* From 0, step to the smallest subnormal magnitude */
        /* Sign is determined by the direction of y         */
        ux.u = 1ULL; 
        if (y < 0.0) ux.u |= 0x8000000000000000ULL;
        return ux.d;
    }

    /* Determine if we need to increase or decrease the absolute magnitude */
    /* x > 0 and x < y  --> Increase magnitude                             */
    /* x < 0 and x > y  --> Increase magnitude (becomes more negative)     */
    if ((x > 0.0) == (x < y))
    {
        ux.u++; /* Incrementing moves magnitude away from 0 */
    } else {
        ux.u--; /* Decrementing moves magnitude toward 0    */
    }

    return ux.d;
}

static inline i64
f64_ulp_distance(f64 a, f64 b, int64_t max_threshold) 
{

    /* Handle exact equality (including -0.0 and +0.0, which are 0 ULPs apart) */
    if (a == b || (eagle_isnan(a) && eagle_isnan(b))) {
        return 0;
    }

    /* Count steps using nextafter */
    int64_t ulps = 0;
    f64 current = a;

    while (current != b && ulps <= max_threshold) {
        current = eagle_nextafter(current, b);
        ulps++;
    }

    return ulps;
}

static inline b32 
eagle_isnan_bits(f64 x)
{
    union { f64 d; u64 u; } pun;
    
    pun.d = x;
    /* Exponent mask: 0x7FF0000000000000ULL and Mantissa mask: 0x000FFFFFFFFFFFFFULL */
    return ((pun.u & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) & ((pun.u & 0x000FFFFFFFFFFFFFULL) != 0ULL);
}


static inline f64
eagle_nearbyint(f64 x)
{
    /* 2^52 represents the threshold where a f64 no longer has fractional bits */
    f64 const magic = +0x1.0000000000000p+0052;
    
    /* Handle negative numbers by tracking the sign bit */
    /* This ensures correct behavior and preserves signed zeros (-0.0) */
    f64 abs_x = (x < 0.0) ? -x : x;

    if (abs_x < magic)
    {
        f64 r = abs_x + magic;
        r -= magic;
        return (x < 0.0) ? -r : r;
    }
    
    /* If x is already larger than 2^52, it has no fractional part, 
       or it is already an Inf/NaN. Return it unmodified. */
    return x;
}

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

    /* Decompose into a number in the range [1.0, 2.0) and the exponent. */
    u64 mantissa = pun.bits & 0x000FFFFFFFFFFFFFLL;
    u64 raw_exponent = (pun.bits >> 52) & 0x7FF;
    i32 exponent = (i32)(raw_exponent - 1023);
    significand.bits = (1023ULL << 52) | mantissa;

    /* Map from [1.0, 2.0) to [sqrt(2)/2, sqrt(2)) */
    f64 s = significand.x;
    i32 adj = (s >= SQRT2) - (s < SQRT2_2);  /* +1, 0, or -1 */
    exponent += adj;
    s *= (adj == 0) ? 1.0 : (adj > 0 ? 0.5 : 2.0);

    f64 const z = s - 1.0;

    f64 const c00 =  0x1.0000000000105p0;
    f64 const c01 =  -0x1.000000000389fp-1;
    f64 const c02 =  0x1.555555544dedap-2;
    f64 const c03 =  -0x1.ffffffeadbebbp-3;
    f64 const c04 =  0x1.99999aea7dee8p-3;
    f64 const c05 =  -0x1.55555efc39788p-3;
    f64 const c06 =  0x1.2491fbb2fd97fp-3;
    f64 const c07 =  -0x1.fffbfb45ae0b1p-4;
    f64 const c08 =  0x1.c72c0284a341ep-4;
    f64 const c09 =  -0x1.9a08da08f5387p-4;
    f64 const c10 =  0x1.73be651915b18p-4;
    f64 const c11 =  -0x1.4ef69dc80b34cp-4;
    f64 const c12 =  0x1.3a48d03bebec3p-4;
    f64 const c13 =  -0x1.517f3303e417cp-4;
    f64 const c14 =  0x1.48373d6644ffbp-4;
    f64 const c15 =  -0x1.4679a50504a83p-5;

    f64 sum = c15;
    sum = __builtin_fma(sum, z, c14);
    sum = __builtin_fma(sum, z, c13);
    sum = __builtin_fma(sum, z, c12);
    sum = __builtin_fma(sum, z, c11);
    sum = __builtin_fma(sum, z, c10);
    sum = __builtin_fma(sum, z, c09);
    sum = __builtin_fma(sum, z, c08);
    sum = __builtin_fma(sum, z, c07);
    sum = __builtin_fma(sum, z, c06);
    sum = __builtin_fma(sum, z, c05);
    sum = __builtin_fma(sum, z, c04);
    sum = __builtin_fma(sum, z, c03);
    sum = __builtin_fma(sum, z, c02);
    sum = __builtin_fma(sum, z, c01);
    sum = __builtin_fma(sum, z, c00);
    sum = __builtin_fma(sum, z, 0.0);

    f64 const LN2_HI = +0x1.62e42fefa39efp-1;
    f64 const LN2_LO = +0x1.abc9e3b39803f8p-56;

    f64 const exp_term_hi = (f64)exponent * LN2_HI;
    f64 const exp_term_lo = (f64)exponent * LN2_LO;

    f64 const tmp = exp_term_hi + sum;
    f64 const err = (exp_term_hi - tmp) + sum;   // error from first addition
    f64 const result = tmp + (err + exp_term_lo);

    return result;

#undef SQRT2
#undef SQRT2_2
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
#define LN2_HI  _mm256_set1_pd(+0x1.62e42fefa39efp-0001)
#define LN2_LO  _mm256_set1_pd(+0x1.abc9e3b39803f8p-0056)

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

    /* Polynomial approximation to log(1 + r) */
    __m256d const c00 = _mm256_set1_pd(0x1.0000000000105p0);
    __m256d const c01 = _mm256_set1_pd(-0x1.000000000389fp-1);
    __m256d const c02 = _mm256_set1_pd(0x1.555555544dedap-2);
    __m256d const c03 = _mm256_set1_pd(-0x1.ffffffeadbebbp-3);
    __m256d const c04 = _mm256_set1_pd(0x1.99999aea7dee8p-3);
    __m256d const c05 = _mm256_set1_pd(-0x1.55555efc39788p-3);
    __m256d const c06 = _mm256_set1_pd(0x1.2491fbb2fd97fp-3);
    __m256d const c07 = _mm256_set1_pd(-0x1.fffbfb45ae0b1p-4);
    __m256d const c08 = _mm256_set1_pd(0x1.c72c0284a341ep-4);
    __m256d const c09 = _mm256_set1_pd(-0x1.9a08da08f5387p-4);
    __m256d const c10 = _mm256_set1_pd(0x1.73be651915b18p-4);
    __m256d const c11 = _mm256_set1_pd(-0x1.4ef69dc80b34cp-4);
    __m256d const c12 = _mm256_set1_pd(0x1.3a48d03bebec3p-4);
    __m256d const c13 = _mm256_set1_pd(-0x1.517f3303e417cp-4);
    __m256d const c14 = _mm256_set1_pd(0x1.48373d6644ffbp-4);
    __m256d const c15 = _mm256_set1_pd(-0x1.4679a50504a83p-5);

    __m256d sum = c15;
    sum = _mm256_fmadd_pd(sum, significand, c14);
    sum = _mm256_fmadd_pd(sum, significand, c13);
    sum = _mm256_fmadd_pd(sum, significand, c12);
    sum = _mm256_fmadd_pd(sum, significand, c11);
    sum = _mm256_fmadd_pd(sum, significand, c10);
    sum = _mm256_fmadd_pd(sum, significand, c09);
    sum = _mm256_fmadd_pd(sum, significand, c08);
    sum = _mm256_fmadd_pd(sum, significand, c07);
    sum = _mm256_fmadd_pd(sum, significand, c06);
    sum = _mm256_fmadd_pd(sum, significand, c05);
    sum = _mm256_fmadd_pd(sum, significand, c04);
    sum = _mm256_fmadd_pd(sum, significand, c03);
    sum = _mm256_fmadd_pd(sum, significand, c02);
    sum = _mm256_fmadd_pd(sum, significand, c01);
    sum = _mm256_fmadd_pd(sum, significand, c00);
    sum = _mm256_mul_pd(sum, significand);

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
    const __m512d LN2_HI    = _mm512_set1_pd(0x1.62e42fefa39efp-1);
    const __m512d LN2_LO    = _mm512_set1_pd(0x1.abc9e3b39803f8p-56);

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

    r = _mm512_sub_pd(r, ONE);   // argument for log1p

    /* High-degree polynomial for log1p(r) */
    __m512d c15 = _mm512_set1_pd(-0x1.4679a50504a83p-5);
    __m512d c14 = _mm512_set1_pd( 0x1.48373d6644ffbp-4);
    __m512d c13 = _mm512_set1_pd(-0x1.517f3303e417cp-4);
    __m512d c12 = _mm512_set1_pd( 0x1.3a48d03bebec3p-4);
    __m512d c11 = _mm512_set1_pd(-0x1.4ef69dc80b34cp-4);
    __m512d c10 = _mm512_set1_pd( 0x1.73be651915b18p-4);
    __m512d c09 = _mm512_set1_pd(-0x1.9a08da08f5387p-4);
    __m512d c08 = _mm512_set1_pd( 0x1.c72c0284a341ep-4);
    __m512d c07 = _mm512_set1_pd(-0x1.fffbfb45ae0b1p-4);
    __m512d c06 = _mm512_set1_pd( 0x1.2491fbb2fd97fp-3);
    __m512d c05 = _mm512_set1_pd(-0x1.55555efc39788p-3);
    __m512d c04 = _mm512_set1_pd( 0x1.99999aea7dee8p-3);
    __m512d c03 = _mm512_set1_pd(-0x1.ffffffeadbebbp-3);
    __m512d c02 = _mm512_set1_pd( 0x1.555555544dedap-2);
    __m512d c01 = _mm512_set1_pd(-0x1.000000000389fp-1);
    __m512d c00 = _mm512_set1_pd( 0x1.0000000000105p+0);

    __m512d p = c15;
    p = _mm512_fmadd_pd(p, r, c14);
    p = _mm512_fmadd_pd(p, r, c13);
    p = _mm512_fmadd_pd(p, r, c12);
    p = _mm512_fmadd_pd(p, r, c11);
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

    __m512d log1p_term = _mm512_mul_pd(p, r);

    /* exponent * ln(2) with accurate summation */
    __m512d exp_pd = _mm512_cvtepi64_pd(exp_adj);
    __m512d exp_hi = _mm512_mul_pd(exp_pd, LN2_HI);
    __m512d exp_lo = _mm512_mul_pd(exp_pd, LN2_LO);

    __m512d tmp = _mm512_add_pd(exp_hi, log1p_term);
    __m512d err = _mm512_add_pd(_mm512_sub_pd(exp_hi, tmp), log1p_term);

    __m512d result = _mm512_add_pd(tmp, _mm512_add_pd(err, exp_lo));

    /* Apply special cases */
    result = _mm512_mask_blend_pd(special_mask, result, _mm512_set1_pd(EAGLE_NAN));
    result = _mm512_mask_blend_pd(posinf_mask, result, _mm512_set1_pd(EAGLE_POS_INF));
    result = _mm512_mask_blend_pd(zero_mask,    result, _mm512_set1_pd(EAGLE_NEG_INF));

    return result;
}

#endif

#endif

