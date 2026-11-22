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
static inline i64 f64_ulp_distance(f64 a, f64 b);
static inline b32 eagle_isnan_bits(f64 x);
static inline f64 eagle_nearbyint(f64 x);
static inline f64 eagle_exp(f64 x);
static inline f64 eagle_expm1(f64 x);
static inline f64 eagle_log(f64 x);
static inline f64 eagle_log1p(f64 x);
static inline f64 eagle_log1mexp(f64 x);

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                                         AVX2
 *
 *-------------------------------------------------------------------------------------------------------------------------*/
#ifdef __AVX2__

static inline __m256d eagle_avx2_exp_pd(__m256d x);
static inline __m256d eagle_avx2_expm1_pd(__m256d x);
static inline __m256d eagle_avx2_log_pd(__m256d x);
static inline __m256d eagle_avx2_log1p_pd(__m256d x);
static inline __m256d eagle_avx2_log1mexp_pd(__m256d x);

#endif

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                                        AVX512
 *
 *-------------------------------------------------------------------------------------------------------------------------*/
#if ELK_AVX_512 

static inline __m512d eagle_avx512_exp_pd(__m512d x);
static inline __m512d eagle_avx512_expm1_pd(__m512d x);
static inline __m512d eagle_avx512_log_pd(__m512d x);
static inline __m512d eagle_avx512_log1p_pd(__m512d x);
static inline __m512d eagle_avx512_log1mexp_pd(__m512d x);

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
f64_ulp_distance(f64 a, f64 b)
{
    if (a == b || (eagle_isnan(a) && eagle_isnan(b))) { return 0; }

    /* Make both numbers positive for the integer view (or handle the sign bit carefully) */
    union { f64 f; i64 i; } ua = {a}, ub = {b};

    /* Two's-complement ordering of IEEE-754 numbers (this is the standard integer-distance trick) */
    if (ua.i < 0) { ua.i = 0x8000000000000000LL - ua.i; }
    if (ub.i < 0) { ub.i = 0x8000000000000000LL - ub.i; }

    i64 dist = ua.i - ub.i;
    return dist < 0 ? -dist : dist;
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

#endif

