#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpfr.h>

#include "elk.h"
#include "magpie.h"
#include "coyote.h"

#include "eagle_test.h"
#include "../exp/exp.c"
#include "../expm1/expm1.c"
#include "../log/log.c"
#include "../log1p/log1p.c"

#include<immintrin.h>

static f64 const min_exp_arg = -710.0;
static f64 const max_exp_arg = 710.0;
static f64 const min_log_arg = -1.0e-10;
static f64 const max_log_arg = 0x1.fffffffffffffp1023;

#define COUNT_TRIALS (8 * 100000000)

void
test_exp_scalar(void)
{
    srand(11);

    f64 min = min_exp_arg;
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 x = min + (f64)rand() / RAND_MAX * (max - min);
        f64 std_math = exp(x);
        f64 eagle_math = eagle_exp(x);

        i64 diff = f64_ulp_distance(std_math, eagle_math);
        ++count;

        if(diff > 0) { ++count1; }

        if(diff > 1)
        {
            printf("scalar input: %a stdlib exp: %a custom exp: %a difference in ulps %ld\n",
                    x, std_math, eagle_math, diff);
        }
    }

    printf("  exp - Scalar |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_expm1_scalar(void)
{
    srand(11);

    f64 min = min_exp_arg;
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 x = min + (f64)rand() / RAND_MAX * (max - min);
        f64 std_math = expm1(x);
        f64 eagle_math = eagle_expm1(x);

        i64 diff = f64_ulp_distance(std_math, eagle_math);
        ++count;

        if(diff > 0) { ++count1; }

        if(diff > 1)
        {
            printf("scalar input: %a stdlib expm1: %a custom expm1: %a difference in ulps %ld\n",
                    x, std_math, eagle_math, diff);
        }
    }

    printf("expm1 - Scalar |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

static inline f64
ulp_error_vs_mpfr(f64 x, f64 my_log)
{
    mpfr_t mx, mlog;
    mpfr_init2(mx, 53);
    mpfr_init2(mlog, 200);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_log(mlog, mx, MPFR_RNDN);          /* high-precision true value */
    mpfr_set_d(mx, my_log, MPFR_RNDN);      /* my result as mpfr         */
    mpfr_sub(mlog, mlog, mx, MPFR_RNDN);    /* true - my                 */

    /* Convert absolute difference into ulps of the double result ulp = 2^(mpfr_exp(mx) - 53) */
    mpfr_mul_2si(mlog, mlog, 53 - mpfr_get_exp(mx), MPFR_RNDN);

    f64 err = mpfr_get_d(mlog, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(mlog);
    return err;
}

void
test_log_scalar(void)
{
    srand(11);

    f64 const min = min_log_arg;
    f64 const mid1 = 1.0;
    f64 const mid2 = 2.0;
    f64 const max = max_log_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,         COUNT_TRIALS };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            f64 x = minr + (f64)rand() / RAND_MAX * (maxr - minr);
            f64 std_math = log(x);
            f64 eagle_math = eagle_log(x);

            i64 diff = f64_ulp_distance(std_math, eagle_math);
            ++count;

            if(diff > 0)
            {
                f64 mpfr_diff = ulp_error_vs_mpfr(x, eagle_math);
                diff = (i64)nearbyint(mpfr_diff);
                if(diff > 0) { ++count1; }

                if(diff > 1)
                {
                    printf("scalar input: %a stdlib log: %a log: %a diff in ulps %ld (%.2lf - mpfr)\n",
                            x, std_math, eagle_math, diff, mpfr_diff);
                }
            }
        }
    }

    printf("  log - Scalar |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log1p_scalar(void)
{
     srand(11);

    f64 min = min_log_arg - 1.0;
    f64 max = max_log_arg - 1.0;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; ++i)
    {

        f64 x = min + (f64)rand() / RAND_MAX * (max - min);
        f64 std_math = log1p(x);
        f64 eagle_math = eagle_log1p(x);

        i64 diff = f64_ulp_distance(std_math, eagle_math);
        ++count;

        if(diff > 0) { ++count1; }

        if(diff > 1)
        {
            printf("scalar input: %a stdlib log1p: %a custom log1p: %a difference in ulps %ld\n",
                    x, std_math, eagle_math, diff);
        }
    }

    printf("log1p - Scalar |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

#ifdef __AVX2__ 

typedef union
{
    __m256d vec;
    f64 arr[4];
} Vec256Pun;

void
test_exp_avx2(void)
{
    srand(11);

    f64 min = min_exp_arg;
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; i += 4)
    {
        Vec256Pun x = {0};
        Vec256Pun eagle_math = {0};
        f64 std_math[4] ={0};

        for(i32 j = 0; j < 4; ++j)
        {
            x.arr[j] = min + (f64)rand() / RAND_MAX * (max - min);
            std_math[j] = exp(x.arr[j]);
        }

        eagle_math.vec = eagle_avx2_exp_pd(x.vec);

        for(i32 j = 0; j < 4; ++j)
        {
            i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                printf("avx2 input: %a stdlib exp: %a custom exp: %a difference in ulps %ld\n",
                        x.arr[j], std_math[j], eagle_math.arr[j], diff);
            }
        }
    }

    printf("  exp - AVX2   |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_expm1_avx2(void)
{
    srand(11);

    f64 min = min_exp_arg;
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; i += 4)
    {
        Vec256Pun x = {0};
        Vec256Pun eagle_math = {0};
        f64 std_math[4] ={0};

        for(i32 j = 0; j < 4; ++j)
        {
            x.arr[j] = min + (f64)rand() / RAND_MAX * (max - min);
            std_math[j] = expm1(x.arr[j]);
        }

        eagle_math.vec = eagle_avx2_expm1_pd(x.vec);

        for(i32 j = 0; j < 4; ++j)
        {
            i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                printf("avx2 input: %a stdlib expm1: %a custom expm1: %a difference in ulps %ld\n",
                        x.arr[j], std_math[j], eagle_math.arr[j], diff);
            }
        }
    }

    printf("expm1 - AVX2   |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log_avx2(void)
{
    srand(11);

    f64 const min = min_log_arg;
    f64 const mid1 = 1.0;
    f64 const mid2 = 2.0;
    f64 const max = max_log_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,         COUNT_TRIALS };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; i += 4)
        {
            Vec256Pun x = {0};
            Vec256Pun eagle_math = {0};

            f64 std_math[4];

            for(i64 j = 0; j < 4; ++j)
            {
                f64 const minr = mins[range];
                f64 const maxr = maxs[range];

                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
                std_math[j] = log(x.arr[j]);
            }

            eagle_math.vec = eagle_avx2_log_pd(x.vec);

            for(i64 j = 0; j < 4; ++j)
            {
                int64_t diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
                ++count;

                if(diff > 0) { ++count1; }

                if(diff > 1)
                {

                    printf("AVX2  input: %a stdlib log: %a custom log: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], eagle_math.arr[j], diff);
                }
            }
        }
    }

    printf("  log - AVX2   |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log1p_avx2(void)
{
    srand(11);

    f64 min = min_log_arg - 1.0;
    f64 max = max_log_arg - 1.0;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; i += 4)
    {
        Vec256Pun x = {0};
        Vec256Pun eagle_math = {0};

        f64 std_math[4];

        for(i64 j = 0; j < 4; ++j)
        {
            x.arr[j] = min + (f64)rand() / RAND_MAX * (max - min);
            std_math[j] = log1p(x.arr[j]);
        }

        eagle_math.vec = eagle_avx2_log1p_pd(x.vec);

        for(i64 j = 0; j < 4; ++j)
        {
            int64_t diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                printf("AVX2  input: %a stdlib log1p: %a custom log1p: %a difference in ulps %ld\n",
                        x.arr[j], std_math[j], x.arr[j], diff);
            }
        }
    }

    printf("log1p - AVX2   |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)

typedef union
{
    __m512d vec;
    f64 arr[8];
} Vec512Pun;

void
test_exp_avx512(void)
{
    srand(11);

    f64 min = min_exp_arg;
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; i += 8)
    {

        Vec512Pun x = {0};
        Vec512Pun eagle_math = {0};
        f64 std_math[8];

        for(i64 j = 0; j < 8; ++j)
        {
            f64 val = min + (f64)rand() / RAND_MAX * (max - min);
            x.arr[j] = val;
            std_math[j] = exp(val);
        }

        eagle_math.vec = eagle_avx512_exp_pd(x.vec);

        for(i64 j = 0; j < 8; ++j)
        {
            i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                printf("AVX512 input: %a stdlib exp: %a custom exp: %a difference in ulps %ld\n",
                        x.arr[j], std_math[j], eagle_math.arr[j], diff);
            }
        }
    }

    printf("  exp - AVX512 |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_expm1_avx512(void)
{
    srand(11);

    f64 min = min_exp_arg;
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; i += 8)
    {

        Vec512Pun x = {0};
        Vec512Pun eagle_math = {0};
        f64 std_math[8];

        for(i64 j = 0; j < 8; ++j)
        {
            f64 val = min + (f64)rand() / RAND_MAX * (max - min);
            x.arr[j] = val;
            std_math[j] = expm1(val);
        }

        eagle_math.vec = eagle_avx512_expm1_pd(x.vec);

        for(i64 j = 0; j < 8; ++j)
        {
            i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                printf("AVX512 input: %a stdlib expm1: %a custom expm1: %a difference in ulps %ld\n",
                        x.arr[j], std_math[j], eagle_math.arr[j], diff);
            }
        }
    }

    printf("expm1 - AVX512 |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log_avx512(void)
{
    srand(11);

    f64 const min = min_log_arg;
    f64 const mid1 = 1.0;
    f64 const mid2 = 2.0;
    f64 const max = max_log_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,         COUNT_TRIALS };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; i += 8)
        {
            Vec512Pun x = {0};
            Vec512Pun eagle_math = {0};

            f64 std_math[8];

            for(i64 j = 0; j < 8; ++j)
            {
                f64 const minr = mins[range];
                f64 const maxr = maxs[range];

                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
                std_math[j] = log(x.arr[j]);
            }

            eagle_math.vec = eagle_avx512_log_pd(x.vec);

            for(i64 j = 0; j < 8; ++j)
            {
                i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
                ++count;

                if(diff > 0) { ++count1; }

                if(diff > 1)
                {
                    printf("AVX512 input: %a stdlib log: %a custom log: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], x.arr[j], diff);
                }
            }
        }
    }

    printf("  log - AVX512 |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log1p_avx512(void)
{
    srand(11);

    f64 min = min_log_arg - 1.0;
    f64 max = max_log_arg - 1.0;

    i64 count = 0;
    i64 count1 = 0;

    for(i64 i = 0; i < COUNT_TRIALS; i += 8)
    {
        Vec512Pun x = {0};
        Vec512Pun eagle_math = {0};

        f64 std_math[8];

        for(i64 j = 0; j < 8; ++j)
        {
            x.arr[j] = min + (f64)rand() / RAND_MAX * (max - min);
            std_math[j] = log1p(x.arr[j]);
        }

        eagle_math.vec = eagle_avx512_log1p_pd(x.vec);

        for(i64 j = 0; j < 8; ++j)
        {
            i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                printf("AVX512 input: %a stdlib log1p: %a custom log1p: %a difference in ulps %ld\n",
                        x.arr[j], std_math[j], x.arr[j], diff);
            }
        }
    }

    printf("log1p - AVX512 |  %.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

#endif

int
main(int argc, char *argv[])
{
    coy_profile_begin();
    CoyProfileAnchor ap;

#if 1
    ap = COY_START_PROFILE_BLOCK("log-scalar-test");
    test_log_scalar();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("log1p-scalar-test");
    test_log1p_scalar();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("exp-scalar-test");
    test_exp_scalar();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("expm1-scalar-test");
    test_expm1_scalar();
    COY_END_PROFILE(ap);
#endif

#if 1 && defined(__AVX2__ ) 

    ap = COY_START_PROFILE_BLOCK("log-avx2-test");
    test_log_avx2();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("log1p-avx2-test");
    test_log1p_avx2();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("exp-avx2-test");
    test_exp_avx2();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("expm1-avx2-test");
    test_expm1_avx2();
    COY_END_PROFILE(ap);

#endif

#if 1 && defined(__AVX512F__) && defined(__AVX512BW__)

    ap = COY_START_PROFILE_BLOCK("log-avx512-test");
    test_log_avx512();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("log1p-avx512-test");
    test_log1p_avx512();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("exp-avx512-test");
    test_exp_avx512();
    COY_END_PROFILE(ap);

    ap = COY_START_PROFILE_BLOCK("expm1-avx512-test");
    test_expm1_avx512();
    COY_END_PROFILE(ap);

#endif

    coy_profile_end();

#if COY_PROFILE
    printf("\n\nTotal Runtime = %.3lf seconds at a frequency of %"PRIu64"\n",
            coy_global_profiler.total_elapsed, coy_global_profiler.freq);

    u64 log_scalar_elapsed = 0;
    u64 log_avx2_elapsed = 0;
    u64 log_avx512_elapsed = 0;

    u64 log1p_scalar_elapsed = 0;
    u64 log1p_avx2_elapsed = 0;
    u64 log1p_avx512_elapsed = 0;

    u64 exp_scalar_elapsed = 0;
    u64 exp_avx2_elapsed = 0;
    u64 exp_avx512_elapsed = 0;

    u64 expm1_scalar_elapsed = 0;
    u64 expm1_avx2_elapsed = 0;
    u64 expm1_avx512_elapsed = 0;

    for(i32 i = 0; i < COY_PROFILE_NUM_BLOCKS; ++i)
    {
        CoyBlockProfiler *block = &coy_global_profiler.blocks[i];
        if(block->hit_count)
        {
            ElkStr label = elk_str_from_cstring((char *)block->label);

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-scalar-test", .len = 17}))
            {
                log1p_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-scalar-test", .len = 15}))
            {
                log_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-scalar-test", .len = 15}))
            {
                exp_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "expm1-scalar-test", .len = 17}))
            {
                expm1_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-avx2-test", .len = 15}))
            {
                log1p_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-avx2-test", .len = 13}))
            {
                log_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-avx2-test", .len = 13}))
            {
                exp_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "expm1-avx2-test", .len = 15}))
            {
                expm1_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-avx512-test", .len = 17}))
            {
                log1p_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-avx512-test", .len = 15}))
            {
                log_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-avx512-test", .len = 15}))
            {
                exp_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "expm1-avx512-test", .len = 17}))
            {
                expm1_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            printf("%-32s Hits: %3"PRIu64" Exclusive: %6.2lf%%", block->label, block->hit_count, block->exclusive_pct);
            
            if(block->inclusive_pct != block->exclusive_pct)
            {
                printf(" Inclusive: %6.2lf%%\n", block->inclusive_pct);
            }
            else
            {
                printf("\n");
            }
        }
    }

    printf("\n");
    printf("  exp() AVX2 Speed up   = %.2lf\n", (f64)exp_scalar_elapsed / (f64)exp_avx2_elapsed);
    printf("  exp() AVX512 Speed up = %.2lf\n", (f64)exp_scalar_elapsed / (f64)exp_avx512_elapsed);
    printf("expm1() AVX2 Speed up   = %.2lf\n", (f64)expm1_scalar_elapsed / (f64)expm1_avx2_elapsed);
    printf("expm1() AVX512 Speed up = %.2lf\n", (f64)expm1_scalar_elapsed / (f64)expm1_avx512_elapsed);
    printf("  log() AVX2 Speed up   = %.2lf\n", (f64)log_scalar_elapsed / (f64)log_avx2_elapsed);
    printf("  log() AVX512 Speed up = %.2lf\n", (f64)log_scalar_elapsed / (f64)log_avx512_elapsed);
    printf("log1p() AVX2 Speed up   = %.2lf\n", (f64)log1p_scalar_elapsed / (f64)log1p_avx2_elapsed);
    printf("log1p() AVX512 Speed up = %.2lf\n", (f64)log1p_scalar_elapsed / (f64)log1p_avx512_elapsed);

#endif
    return EXIT_SUCCESS;
}

COY_PROFILE_STATIC_CHECK ;
