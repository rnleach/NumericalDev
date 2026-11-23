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
#include "../log1mexp/log1mexp.c"

#include <immintrin.h>

_Static_assert(DBL_HAS_SUBNORM == 1, "Subnormal numbers not supported.");

static f64 const min_exp_arg = -0x1.74910d52d3050p+0009;
static f64 const max_exp_arg = +0x1.62e42fefa39efp+0009;

#define COUNT_TRIALS (8LL * 100000000LL)

void
test_exp_scalar(void)
{
    /* Test special cases */
    f64 test_result;
    test_result = eagle_exp(EAGLE_NAN);         Assert(isnan(test_result));
    test_result = eagle_exp(EAGLE_POS_INF);     Assert(test_result == EAGLE_POS_INF);
    test_result = eagle_exp(max_exp_arg + 1.0); Assert(test_result == EAGLE_POS_INF);
    test_result = eagle_exp(EAGLE_NEG_INF);     Assert(test_result == 0.0);
    test_result = eagle_exp(min_exp_arg - 1.0); Assert(test_result == 0.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 min = min_exp_arg;          /* Negative inputs with representable outputs. */
    f64 mid1 = -EAGLE_SUBNORM_MAX;  /* Subnormals.                                 */
    f64 mid2 = EAGLE_SUBNORM_MAX;   /* Positive inputs with representable outputs. */
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,     COUNT_TRIALS     };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("exp-scalar-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            f64 x = minr + (f64)rand() / RAND_MAX * (maxr - minr);
            f64 std_math = exp(x);
            f64 eagle_math = eagle_exp(x);

            i64 diff = f64_ulp_distance(std_math, eagle_math);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                fprintf(stderr, "scalar input: %a stdlib exp: %a custom exp: %a difference in ulps %ld\n",
                        x, std_math, eagle_math, diff);
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "     exp - Scalar |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

static inline f64
ulp_error_vs_mpfr_log(f64 x, f64 my_log)
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
    /* Test special cases */
    f64 test_result;
    test_result = eagle_log(EAGLE_NAN);     Assert(isnan(test_result));
    test_result = eagle_log(-1.0);          Assert(isnan(test_result));
    test_result = eagle_log(0.0);           Assert(test_result == EAGLE_NEG_INF);
    test_result = eagle_log(EAGLE_POS_INF); Assert(test_result == EAGLE_POS_INF);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals                  */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Subnormal to reduced range. */
    f64 const mid2 = sqrt(2.0) / 2.0;   /* Reduced range.              */
    f64 const mid3 = sqrt(2.0);         /* Reduce range to max double  */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[4]  = {                0, COUNT_TRIALS / 4,     COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4 };
    i64 const highs[4] = { COUNT_TRIALS / 4, COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4,     COUNT_TRIALS     };
    f64 const mins[4]  = {  min, mid1, mid2, mid3 };
    f64 const maxs[4]  = { mid1, mid2, mid3,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log-scalar-test");
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
                f64 mpfr_diff = ulp_error_vs_mpfr_log(x, eagle_math);
                diff = (i64)nearbyint(mpfr_diff);
                if(diff > 0) { ++count1; }

                if(diff > 1)
                {
                    fprintf(stderr, "scalar input: %a stdlib log: %a log: %a diff in ulps %ld (%.2lf - mpfr)\n",
                            x, std_math, eagle_math, diff, mpfr_diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "     log - Scalar |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_expm1_scalar(void)
{
    /* Test special cases */
    f64 test_result;
    test_result = eagle_expm1(EAGLE_NAN);         Assert(isnan(test_result));
    test_result = eagle_expm1(EAGLE_POS_INF);     Assert(test_result == EAGLE_POS_INF);
    test_result = eagle_expm1(max_exp_arg + 1.0); Assert(test_result == EAGLE_POS_INF);
    test_result = eagle_expm1(EAGLE_NEG_INF);     Assert(test_result == -1.0);
    test_result = eagle_expm1(min_exp_arg - 1.0); Assert(test_result == -1.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 min = min_exp_arg;          /* Negative inputs with representable outputs. */
    f64 mid1 = -EAGLE_SUBNORM_MAX;  /* Subnormals.                                 */
    f64 mid2 = EAGLE_SUBNORM_MAX;   /* Positive inputs with representable outputs. */
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,     COUNT_TRIALS     };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("expm1-scalar-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            f64 x = minr + (f64)rand() / RAND_MAX * (maxr - minr);
            f64 std_math = expm1(x);
            f64 eagle_math = eagle_expm1(x);

            i64 diff = f64_ulp_distance(std_math, eagle_math);
            ++count;

            if(diff > 0) { ++count1; }

            if(diff > 1)
            {
                fprintf(stderr, "scalar input: %a stdlib expm1: %a custom expm1: %a difference in ulps %ld\n",
                        x, std_math, eagle_math, diff);
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "   expm1 - Scalar |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

static inline f64
ulp_error_vs_mpfr_log1p(f64 x, f64 my_log1p)
{
    mpfr_t mx, mlog1p;
    mpfr_init2(mx, 53);
    mpfr_init2(mlog1p, 200);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_log(mlog1p, mx, MPFR_RNDN);            /* high-precision true value */
    mpfr_set_d(mx, my_log1p, MPFR_RNDN);        /* my result as mpfr         */
    mpfr_sub(mlog1p, mlog1p, mx, MPFR_RNDN);    /* true - my                 */

    /* Convert absolute difference into ulps of the double result ulp = 2^(mpfr_exp(mx) - 53) */
    mpfr_mul_2si(mlog1p, mlog1p, 53 - mpfr_get_exp(mx), MPFR_RNDN);

    f64 err = mpfr_get_d(mlog1p, MPFR_RNDN);

    mpfr_clear(mx);
    mpfr_clear(mlog1p);
    return err;
}

void
test_log1p_scalar(void)
{
    /* Test special cases */
    f64 test_result;
    test_result = eagle_log1p(EAGLE_NAN);     Assert(isnan(test_result));
    test_result = eagle_log1p(-1.01);         Assert(isnan(test_result));
    test_result = eagle_log1p(-1.0);          Assert(test_result == EAGLE_NEG_INF);
    test_result = eagle_log1p(EAGLE_POS_INF); Assert(test_result == EAGLE_POS_INF);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals                  */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Subnormal to reduced range. */
    f64 const mid2 = sqrt(2.0) / 2.0;   /* Reduced range.              */
    f64 const mid3 = sqrt(2.0);         /* Reduce range to max double  */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[4]  = {                0, COUNT_TRIALS / 4,     COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4 };
    i64 const highs[4] = { COUNT_TRIALS / 4, COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4,     COUNT_TRIALS     };
    f64 const mins[4]  = {  min, mid1, mid2, mid3 };
    f64 const maxs[4]  = { mid1, mid2, mid3,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1p-scalar-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            f64 x = minr + (f64)rand() / RAND_MAX * (maxr - minr);
            f64 std_math = log1p(x);
            f64 eagle_math = eagle_log1p(x);

            i64 diff = f64_ulp_distance(std_math, eagle_math);
            ++count;

            if(diff > 0) 
            {
                f64 mpfr_diff = ulp_error_vs_mpfr_log1p(x, eagle_math);
                diff = (i64)nearbyint(mpfr_diff);
                if(diff > 0) { ++count1; }

                if(diff > 1)
                {
                    fprintf(stderr, "scalar input: %a stdlib log1p: %a custom log1p: %a difference in ulps %ld\n",
                            x, std_math, eagle_math, diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "   log1p - Scalar |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

static inline f64
ulp_error_vs_mpfr_log1mexp(double x, double my_log1mexp)
{
    mpfr_t mx, neg_x, one_minus_exp, truth, mine, diff;
    mpfr_init2(mx, 53);
    mpfr_init2(neg_x, 256);
    mpfr_init2(one_minus_exp, 256);
    mpfr_init2(truth, 256);
    mpfr_init2(mine, 53);
    mpfr_init2(diff, 256);

    mpfr_set_d(mx, x, MPFR_RNDN);
    mpfr_neg(neg_x, mx, MPFR_RNDN);
    /* 1 - exp(-x) = -expm1(-x) */
    mpfr_expm1(one_minus_exp, neg_x, MPFR_RNDN);
    mpfr_neg(one_minus_exp, one_minus_exp, MPFR_RNDN);
    mpfr_log(truth, one_minus_exp, MPFR_RNDN);

    mpfr_set_d(mine, my_log1mexp, MPFR_RNDN);
    mpfr_sub(diff, truth, mine, MPFR_RNDN);

    mpfr_exp_t exp2 = mpfr_zero_p(mine) ? -1022 : mpfr_get_exp(mine);
    if(exp2 < -1021)
    {
        exp2 = -1021; /* subnormal binary64 ULP is 2^-1074 = 2^(-53 - (-1021)) */
    }
    mpfr_mul_2si(diff, diff, 53 - exp2, MPFR_RNDN);

    f64 err = mpfr_get_d(diff, MPFR_RNDN);

    mpfr_clears(mx, neg_x, one_minus_exp, truth, mine, diff, (mpfr_ptr)0);
    return err;
}

void
test_log1mexp_scalar(void)
{
    /* Test special cases */
    f64 test_result;
    test_result = eagle_log1mexp(EAGLE_NAN);     Assert(isnan(test_result));
    test_result = eagle_log1mexp(-1.0);          Assert(isnan(test_result));
    test_result = eagle_log1mexp(0.0);           Assert(test_result == EAGLE_NEG_INF);
    test_result = eagle_log1mexp(EAGLE_POS_INF); Assert(test_result == 0.0);

    /* Seed random number generator for reproducible results */
     srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals.        */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Rest of the range. */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[2]  = {                0, COUNT_TRIALS / 4 };
    i64 const highs[2] = { COUNT_TRIALS / 4, COUNT_TRIALS     };
    f64 const mins[2]  = {  min, mid1 };
    f64 const maxs[2]  = { mid1,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1mexp-scalar-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            f64 x = minr + (f64)rand() / RAND_MAX * (maxr - minr);
            f64 std_math = test_std_log1mexp(x);
            f64 eagle_math = eagle_log1mexp(x);

            i64 diff = f64_ulp_distance(std_math, eagle_math);
            ++count;

            if(diff > 0) 
            {
                f64 mpfr_diff = ulp_error_vs_mpfr_log1mexp(x, eagle_math);
                diff = (i64)nearbyint(mpfr_diff);
                if(diff > 0) { ++count1; }

                if(diff > 1)
                {
                    fprintf(stderr, "scalar input: %a stdlib log1mexp: %a custom log1mexp: %a difference in ulps %ld\n",
                            x, std_math, eagle_math, diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "log1mexp - Scalar |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
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
    /* Test special cases */
    Vec256Pun test_x = {0};
    Vec256Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = EAGLE_POS_INF;
    test_x.arr[2] = max_exp_arg + 1.0;
    test_x.arr[3] = EAGLE_NEG_INF;

    test_result.vec = eagle_avx2_exp_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(test_result.arr[1] == EAGLE_POS_INF);
    Assert(test_result.arr[2] == EAGLE_POS_INF);
    Assert(test_result.arr[3] == 0.0);

    test_x.arr[0] = min_exp_arg - 1.0;
    test_result.vec = eagle_avx2_exp_pd(test_x.vec);
    Assert(test_result.arr[0] == 0.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 min = min_exp_arg;          /* Negative inputs with representable outputs. */
    f64 mid1 = -EAGLE_SUBNORM_MAX;  /* Subnormals.                                 */
    f64 mid2 = EAGLE_SUBNORM_MAX;   /* Positive inputs with representable outputs. */
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,     COUNT_TRIALS     };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("exp-avx2-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec256Pun x = {0};
            Vec256Pun eagle_math = {0};
            f64 std_math[4] ={0};

            for(i32 j = 0; j < 4; ++j)
            {
                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
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
                    fprintf(stderr, "avx2 input: %a stdlib exp: %a custom exp: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], eagle_math.arr[j], diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "     exp - AVX2   |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log_avx2(void)
{
    /* Test special cases */
    Vec256Pun test_x = {0};
    Vec256Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = -1.0;
    test_x.arr[2] = 0.0;
    test_x.arr[3] = EAGLE_POS_INF;

    test_result.vec = eagle_avx2_log_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(isnan(test_result.arr[1]));
    Assert(test_result.arr[2] == EAGLE_NEG_INF);
    Assert(test_result.arr[3] == EAGLE_POS_INF);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals                  */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Subnormal to reduced range. */
    f64 const mid2 = sqrt(2.0) / 2.0;   /* Reduced range.              */
    f64 const mid3 = sqrt(2.0);         /* Reduce range to max double  */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[4]  = {                0, COUNT_TRIALS / 4,     COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4 };
    i64 const highs[4] = { COUNT_TRIALS / 4, COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4,     COUNT_TRIALS     };
    f64 const mins[4]  = {  min, mid1, mid2, mid3 };
    f64 const maxs[4]  = { mid1, mid2, mid3,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log-avx2-test");
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

                    fprintf(stderr, "AVX2  input: %a stdlib log: %a custom log: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], eagle_math.arr[j], diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "     log - AVX2   |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_expm1_avx2(void)
{
    /* Test special cases */
    Vec256Pun test_x = {0};
    Vec256Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = EAGLE_POS_INF;
    test_x.arr[2] = max_exp_arg + 1.0;
    test_x.arr[3] = EAGLE_NEG_INF;

    test_result.vec = eagle_avx2_expm1_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(test_result.arr[1] == EAGLE_POS_INF);
    Assert(test_result.arr[2] == EAGLE_POS_INF);
    Assert(test_result.arr[3] == -1.0);

    test_x.arr[0] = min_exp_arg - 1.0;
    test_result.vec = eagle_avx2_expm1_pd(test_x.vec);
    Assert(test_result.arr[0] == -1.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 min = min_exp_arg;          /* Negative inputs with representable outputs. */
    f64 mid1 = -EAGLE_SUBNORM_MAX;  /* Subnormals.                                 */
    f64 mid2 = EAGLE_SUBNORM_MAX;   /* Positive inputs with representable outputs. */
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,     COUNT_TRIALS     };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("expm1-avx2-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec256Pun x = {0};
            Vec256Pun eagle_math = {0};
            f64 std_math[4] ={0};

            for(i32 j = 0; j < 4; ++j)
            {
                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
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
                    fprintf(stderr, "avx2 input: %a stdlib expm1: %a custom expm1: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], eagle_math.arr[j], diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "   expm1 - AVX2   |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log1p_avx2(void)
{
    /* Test special cases */
    Vec256Pun test_x = {0};
    Vec256Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = -1.01;
    test_x.arr[2] = -1.0;
    test_x.arr[3] = EAGLE_POS_INF;

    test_result.vec = eagle_avx2_log1p_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(isnan(test_result.arr[1]));
    Assert(test_result.arr[2] == EAGLE_NEG_INF);
    Assert(test_result.arr[3] == EAGLE_POS_INF);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals                  */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Subnormal to reduced range. */
    f64 const mid2 = sqrt(2.0) / 2.0;   /* Reduced range.              */
    f64 const mid3 = sqrt(2.0);         /* Reduce range to max double  */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[4]  = {                0, COUNT_TRIALS / 4,     COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4 };
    i64 const highs[4] = { COUNT_TRIALS / 4, COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4,     COUNT_TRIALS     };
    f64 const mins[4]  = {  min, mid1, mid2, mid3 };
    f64 const maxs[4]  = { mid1, mid2, mid3,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1p-avx2-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec256Pun x = {0};
            Vec256Pun eagle_math = {0};

            f64 std_math[4];

            for(i64 j = 0; j < 4; ++j)
            {
                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
                std_math[j] = log1p(x.arr[j]);
            }

            eagle_math.vec = eagle_avx2_log1p_pd(x.vec);

            for(i64 j = 0; j < 4; ++j)
            {
                int64_t diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
                ++count;

                if(diff > 0) 
                {
                    f64 xi = x.arr[j];
                    f64 mpfr_diff = ulp_error_vs_mpfr_log1p(xi, eagle_math.arr[j]);
                    diff = (i64)nearbyint(mpfr_diff);
                    if(diff > 0) { ++count1; }

                    if(diff > 1)
                    {
                        fprintf(stderr, "AVX2  input: %a stdlib log1p: %a custom log1p: %a difference in ulps %ld\n",
                                x.arr[j], std_math[j], eagle_math.arr[j], diff);
                    }
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "   log1p - AVX2   |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log1mexp_avx2(void)
{
    /* Test special cases */
    Vec256Pun test_x = {0};
    Vec256Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = -1.0;
    test_x.arr[2] = 0.0;
    test_x.arr[3] = EAGLE_POS_INF;

    test_result.vec = eagle_avx2_log1mexp_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(isnan(test_result.arr[1]));
    Assert(test_result.arr[2] == EAGLE_NEG_INF);
    Assert(test_result.arr[3] == 0.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals.        */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Rest of the range. */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[2]  = {                0, COUNT_TRIALS / 4 };
    i64 const highs[2] = { COUNT_TRIALS / 4, COUNT_TRIALS     };
    f64 const mins[2]  = {  min, mid1 };
    f64 const maxs[2]  = { mid1,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1mexp-avx2-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec256Pun x = {0};
            Vec256Pun eagle_math = {0};

            f64 std_math[4];

            for(i64 j = 0; j < 4; ++j)
            {
                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
                std_math[j] = eagle_log1mexp(x.arr[j]);
            }

            eagle_math.vec = eagle_avx2_log1mexp_pd(x.vec);

            for(i64 j = 0; j < 4; ++j)
            {
                int64_t diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
                ++count;

                if(diff > 0) 
                {
                    f64 mpfr_diff = ulp_error_vs_mpfr_log1mexp(x.arr[j], eagle_math.arr[j]);
                    diff = (i64)nearbyint(mpfr_diff);
                    if(diff > 0) { ++count1; }

                    if(diff > 1)
                    {
                        fprintf(stderr, "AVX2  input: %a scalar log1mexp: %a custom log1mexp: %a difference in ulps %ld\n",
                                x.arr[j], std_math[j], eagle_math.arr[j], diff);
                    }
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "log1mexp - AVX2   |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
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
    /* Test special cases */
    Vec512Pun test_x = {0};
    Vec512Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = EAGLE_POS_INF;
    test_x.arr[2] = max_exp_arg + 1.0;
    test_x.arr[3] = EAGLE_NEG_INF;
    test_x.arr[4] = min_exp_arg - 1.0;

    test_result.vec = eagle_avx512_exp_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(test_result.arr[1] == EAGLE_POS_INF);
    Assert(test_result.arr[2] == EAGLE_POS_INF);
    Assert(test_result.arr[3] == 0.0);
    Assert(test_result.arr[4] == 0.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 min = min_exp_arg;          /* Negative inputs with representable outputs. */
    f64 mid1 = -EAGLE_SUBNORM_MAX;  /* Subnormals.                                 */
    f64 mid2 = EAGLE_SUBNORM_MAX;   /* Positive inputs with representable outputs. */
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,     COUNT_TRIALS     };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("exp-avx512-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec512Pun x = {0};
            Vec512Pun eagle_math = {0};
            f64 std_math[8];

            for(i64 j = 0; j < 8; ++j)
            {
                f64 val = minr + (f64)rand() / RAND_MAX * (maxr - minr);
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
                    fprintf(stderr, "AVX512 input: %a stdlib exp: %a custom exp: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], eagle_math.arr[j], diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "     exp - AVX512 |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log_avx512(void)
{
    /* Test special cases */
    Vec512Pun test_x = {0};
    Vec512Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = -1.0;
    test_x.arr[2] = 0.0;
    test_x.arr[3] = EAGLE_POS_INF;

    test_result.vec = eagle_avx512_log_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(isnan(test_result.arr[1]));
    Assert(test_result.arr[2] == EAGLE_NEG_INF);
    Assert(test_result.arr[3] == EAGLE_POS_INF);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals                  */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Subnormal to reduced range. */
    f64 const mid2 = sqrt(2.0) / 2.0;   /* Reduced range.              */
    f64 const mid3 = sqrt(2.0);         /* Reduce range to max double  */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[4]  = {                0, COUNT_TRIALS / 4,     COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4 };
    i64 const highs[4] = { COUNT_TRIALS / 4, COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4,     COUNT_TRIALS     };
    f64 const mins[4]  = {  min, mid1, mid2, mid3 };
    f64 const maxs[4]  = { mid1, mid2, mid3,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log-avx512-test");
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
                    fprintf(stderr, "AVX512 input: %a stdlib log: %a custom log: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], eagle_math.arr[j], diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "     log - AVX512 |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_expm1_avx512(void)
{
    /* Test special cases */
    Vec512Pun test_x = {0};
    Vec512Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = EAGLE_POS_INF;
    test_x.arr[2] = max_exp_arg + 1.0;
    test_x.arr[3] = EAGLE_NEG_INF;
    test_x.arr[4] = min_exp_arg - 1.0;

    test_result.vec = eagle_avx512_exp_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(test_result.arr[1] == EAGLE_POS_INF);
    Assert(test_result.arr[2] == EAGLE_POS_INF);
    Assert(test_result.arr[3] == -1.0);
    Assert(test_result.arr[4] == -1.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 min = min_exp_arg;          /* Negative inputs with representable outputs. */
    f64 mid1 = -EAGLE_SUBNORM_MAX;  /* Subnormals.                                 */
    f64 mid2 = EAGLE_SUBNORM_MAX;   /* Positive inputs with representable outputs. */
    f64 max = max_exp_arg;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[3]  = {                0,     COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3 };
    i64 const highs[3] = { COUNT_TRIALS / 3, 2 * COUNT_TRIALS / 3,     COUNT_TRIALS     };
    f64 const mins[3]  = {  min, mid1, mid2 };
    f64 const maxs[3]  = { mid1, mid2,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("expm1-avx512-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec512Pun x = {0};
            Vec512Pun eagle_math = {0};
            f64 std_math[8];

            for(i64 j = 0; j < 8; ++j)
            {
                f64 val = minr + (f64)rand() / RAND_MAX * (maxr - minr);
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
                    fprintf(stderr, "AVX512 input: %a stdlib expm1: %a custom expm1: %a difference in ulps %ld\n",
                            x.arr[j], std_math[j], eagle_math.arr[j], diff);
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "   expm1 - AVX512 |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log1p_avx512(void)
{
    /* Test special cases */
    Vec512Pun test_x = {0};
    Vec512Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = -1.01;
    test_x.arr[2] = -1.0;
    test_x.arr[3] = EAGLE_POS_INF;

    test_result.vec = eagle_avx512_log1p_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(isnan(test_result.arr[1]));
    Assert(test_result.arr[2] == EAGLE_NEG_INF);
    Assert(test_result.arr[3] == EAGLE_POS_INF);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals                  */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Subnormal to reduced range. */
    f64 const mid2 = sqrt(2.0) / 2.0;   /* Reduced range.              */
    f64 const mid3 = sqrt(2.0);         /* Reduce range to max double  */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[4]  = {                0, COUNT_TRIALS / 4,     COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4 };
    i64 const highs[4] = { COUNT_TRIALS / 4, COUNT_TRIALS / 2, 3 * COUNT_TRIALS / 4,     COUNT_TRIALS     };
    f64 const mins[4]  = {  min, mid1, mid2, mid3 };
    f64 const maxs[4]  = { mid1, mid2, mid3,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1p-avx512-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec512Pun x = {0};
            Vec512Pun eagle_math = {0};

            f64 std_math[8];

            for(i64 j = 0; j < 8; ++j)
            {
                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
                std_math[j] = log1p(x.arr[j]);
            }

            eagle_math.vec = eagle_avx512_log1p_pd(x.vec);

            for(i64 j = 0; j < 8; ++j)
            {
                i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
                ++count;

                if(diff > 0)
                {
                    f64 xi = x.arr[j];
                    f64 mpfr_diff = ulp_error_vs_mpfr_log1p(xi, eagle_math.arr[j]);
                    diff = (i64)nearbyint(mpfr_diff);
                    if(diff > 0) { ++count1; }

                    if(diff > 1)
                    {
                        fprintf(stderr, "AVX512 input: %a stdlib log1p: %a custom log1p: %a difference in ulps %ld\n",
                                x.arr[j], std_math[j], eagle_math.arr[j], diff);
                    }
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "   log1p - AVX512 |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

void
test_log1mexp_avx512(void)
{
    /* Test special cases */
    Vec512Pun test_x = {0};
    Vec512Pun test_result = {0};
    test_x.arr[0] = EAGLE_NAN;
    test_x.arr[1] = -1.0;
    test_x.arr[2] = 0.0;
    test_x.arr[3] = EAGLE_POS_INF;

    test_result.vec = eagle_avx512_log1mexp_pd(test_x.vec);
    Assert(isnan(test_result.arr[0]));
    Assert(isnan(test_result.arr[1]));
    Assert(test_result.arr[2] == EAGLE_NEG_INF);
    Assert(test_result.arr[3] == 0.0);

    /* Seed random number generator for reproducible results */
    srand(11);

    /* Set ranges to focus testing on, make sure each range receives adequate testing. */
    f64 const min = EAGLE_SUBNORM_MIN;  /* Subnormals.        */
    f64 const mid1 = EAGLE_SUBNORM_MAX; /* Rest of the range. */
    f64 const max = DBL_MAX;

    i64 count = 0;
    i64 count1 = 0;

    i64 const lows[2]  = {                0, COUNT_TRIALS / 4 };
    i64 const highs[2] = { COUNT_TRIALS / 4, COUNT_TRIALS     };
    f64 const mins[2]  = {  min, mid1 };
    f64 const maxs[2]  = { mid1,  max };

    CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1mexp-avx512-test");
    for(size range = 0; range < ECO_ARRAY_SIZE(lows); ++range)
    {
        for(i64 i = lows[range]; i < highs[range]; ++i)
        {
            f64 const minr = mins[range];
            f64 const maxr = maxs[range];

            Vec512Pun x = {0};
            Vec512Pun eagle_math = {0};

            f64 std_math[8];

            for(i64 j = 0; j < 8; ++j)
            {
                x.arr[j] = minr + (f64)rand() / RAND_MAX * (maxr - minr);
                std_math[j] = eagle_log1mexp(x.arr[j]);
            }

            eagle_math.vec = eagle_avx512_log1mexp_pd(x.vec);

            for(i64 j = 0; j < 8; ++j)
            {
                i64 diff = f64_ulp_distance(std_math[j], eagle_math.arr[j]);
                ++count;

                if(diff > 0) 
                {
                    f64 mpfr_diff = ulp_error_vs_mpfr_log1mexp(x.arr[j], eagle_math.arr[j]);
                    diff = (i64)nearbyint(mpfr_diff);
                    if(diff > 0) { ++count1; }

                    if(diff > 1)
                    {
                        fprintf(stderr, "AVX512  input: %a scalar log1mexp: %a custom log1mexp: %a difference in ulps %ld\n",
                                x.arr[j], std_math[j], eagle_math.arr[j], diff);
                    }
                }
            }
        }
    }

    COY_END_PROFILE(ap);

    fprintf(stdout, "log1mexp - AVX512 |  %5.2lf%% of cases were 1 ULP off\n", (f64)count1 / (f64)count * 100.0);
}

#endif

int
main(int argc, char *argv[])
{
    coy_profile_begin();

#if 1

    test_exp_scalar();
    test_log_scalar();
    test_expm1_scalar();
    test_log1p_scalar();
    test_log1mexp_scalar();

#endif

#if 1 && defined(__AVX2__ ) 

    test_exp_avx2();
    test_log_avx2();
    test_expm1_avx2();
    test_log1p_avx2();
    test_log1mexp_avx2();

#endif

#if 1 && defined(__AVX512F__) && defined(__AVX512BW__)

    test_exp_avx512();
    test_log_avx512();
    test_expm1_avx512();
    test_log1p_avx512();
    test_log1mexp_avx512();

#endif

    coy_profile_end();

#if COY_PROFILE
    fprintf(stdout, "\n\nTotal Runtime = %.3lf seconds at a frequency of %"PRIu64"\n",
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

    u64 log1mexp_scalar_elapsed = 0;
    u64 log1mexp_avx2_elapsed = 0;
    u64 log1mexp_avx512_elapsed = 0;

    for(i32 i = 0; i < COY_PROFILE_NUM_BLOCKS; ++i)
    {
        CoyBlockProfiler *block = &coy_global_profiler.blocks[i];
        if(block->hit_count)
        {
            ElkStr label = elk_str_from_cstring((char *)block->label);

            /* --------------- exp tests --------------- */

            if(elk_str_eq(label, (ElkStr){ .start = "exp-scalar-test", .len = 15}))
            {
                exp_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-avx2-test", .len = 13}))
            {
                exp_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-avx512-test", .len = 15}))
            {
                exp_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            /* --------------- log tests --------------- */

            if(elk_str_eq(label, (ElkStr){ .start = "log-scalar-test", .len = 15}))
            {
                log_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-avx2-test", .len = 13}))
            {
                log_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-avx512-test", .len = 15}))
            {
                log_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            /* --------------- expm1 tests --------------- */

            if(elk_str_eq(label, (ElkStr){ .start = "expm1-scalar-test", .len = 17}))
            {
                expm1_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "expm1-avx2-test", .len = 15}))
            {
                expm1_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "expm1-avx512-test", .len = 17}))
            {
                expm1_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            /* --------------- log1p tests --------------- */

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-scalar-test", .len = 17}))
            {
                log1p_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-avx2-test", .len = 15}))
            {
                log1p_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-avx512-test", .len = 17}))
            {
                log1p_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }


            /* --------------- log1mexp tests --------------- */

            if(elk_str_eq(label, (ElkStr){ .start = "log1mexp-scalar-test", .len = 20}))
            {
                log1mexp_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1mexp-avx2-test", .len = 18}))
            {
                log1mexp_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1mexp-avx512-test", .len = 20}))
            {
                log1mexp_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            /* --------------- end of tests section --------------- */

            fprintf(stdout, "%-32s Hits: %3"PRIu64" Exclusive: %6.2lf%%", block->label, block->hit_count, block->exclusive_pct);
            
            if(block->inclusive_pct != block->exclusive_pct)
            {
                fprintf(stdout, " Inclusive: %6.2lf%%\n", block->inclusive_pct);
            }
            else
            {
                fprintf(stdout, "\n");
            }
        }
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "     exp() AVX2 Speed up   = %.2lf\n", (f64)exp_scalar_elapsed / (f64)exp_avx2_elapsed);
    fprintf(stdout, "     exp() AVX512 Speed up = %.2lf\n", (f64)exp_scalar_elapsed / (f64)exp_avx512_elapsed);
    fprintf(stdout, "   expm1() AVX2 Speed up   = %.2lf\n", (f64)expm1_scalar_elapsed / (f64)expm1_avx2_elapsed);
    fprintf(stdout, "   expm1() AVX512 Speed up = %.2lf\n", (f64)expm1_scalar_elapsed / (f64)expm1_avx512_elapsed);
    fprintf(stdout, "     log() AVX2 Speed up   = %.2lf\n", (f64)log_scalar_elapsed / (f64)log_avx2_elapsed);
    fprintf(stdout, "     log() AVX512 Speed up = %.2lf\n", (f64)log_scalar_elapsed / (f64)log_avx512_elapsed);
    fprintf(stdout, "   log1p() AVX2 Speed up   = %.2lf\n", (f64)log1p_scalar_elapsed / (f64)log1p_avx2_elapsed);
    fprintf(stdout, "   log1p() AVX512 Speed up = %.2lf\n", (f64)log1p_scalar_elapsed / (f64)log1p_avx512_elapsed);
    fprintf(stdout, "log1mexp() AVX2 Speed up   = %.2lf\n", (f64)log1mexp_scalar_elapsed / (f64)log1mexp_avx2_elapsed);
    fprintf(stdout, "log1mexp() AVX512 Speed up = %.2lf\n", (f64)log1mexp_scalar_elapsed / (f64)log1mexp_avx512_elapsed);

#endif
    return EXIT_SUCCESS;
}

COY_PROFILE_STATIC_CHECK ;
