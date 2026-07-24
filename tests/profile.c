#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "elk.h"
#include "magpie.h"
#include "coyote.h"

#include "eagle_test.h"
#include "../exp/exp.c"
#include "../log/log.c"
#include "../log1p/log1p.c"

#include<immintrin.h>

static f64 const min_exp_arg = -710.0;
static f64 const max_exp_arg = 710.0;

static f64 const min_log_arg = -1.0;
static f64 const max_log_arg = 0x1.fffffffffffffp1023;

static f64 const min_log1p_arg = -2.0;
static f64 const max_log1p_arg = 0x1.fffffffffffffp1023;

#define COUNT_TRIALS (8 * 100000000)

static inline void
allocate_buffers_and_fill_xs(f64 min, f64 max, f64 **xs, f64 **vals, MagAllocator *alloc)
{
    *xs = eco_arena_nmalloc(alloc, COUNT_TRIALS, f64);   Assert(*xs);
    *vals = eco_arena_nmalloc(alloc, COUNT_TRIALS, f64); Assert(*vals);

    f64 dx = max - min;

    /* Touch everything so there are no page faults later. */
    for(size i = 0; i < COUNT_TRIALS; ++i)
    {
        (*xs)[i] = (f64)i * dx + min;
        (*vals)[i] = EAGLE_NAN;
    }
}

static inline void
profile_exp_std(f64 *xs, f64 *std_vals)
{
    for(size i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 val = xs[i];
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("exp-std-test");
        f64 eval = exp(val);
        COY_END_PROFILE(ap);
        std_vals[i] = eval;
    }
}

static inline void
profile_log_std(f64 *xs, f64 *std_vals)
{
    for(size i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 val = xs[i];
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log-std-test");
        f64 lval = log(val);
        COY_END_PROFILE(ap);
        std_vals[i] = lval;
    }
}

static inline void
profile_log1p_std(f64 *xs, f64 *std_vals)
{
    for(size i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 val = xs[i];
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1p-std-test");
        f64 lval = log1p(val);
        COY_END_PROFILE(ap);
        std_vals[i] = lval;
    }
}

static inline void
profile_exp_scalar(f64 *xs, f64 *scalar_vals)
{
    for(size i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 val = xs[i];
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("exp-scalar-test");
        f64 eval = eagle_exp(val);
        COY_END_PROFILE(ap);
        scalar_vals[i] = eval;
    }
}

static inline void
profile_log_scalar(f64 *xs, f64 *scalar_vals)
{
    for(size i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 val = xs[i];
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log-scalar-test");
        f64 lval = eagle_log(val);
        COY_END_PROFILE(ap);
        scalar_vals[i] = lval;
    }
}

static inline void
profile_log1p_scalar(f64 *xs, f64 *scalar_vals)
{
    for(size i = 0; i < COUNT_TRIALS; ++i)
    {
        f64 val = xs[i];
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log1p-scalar-test");
        f64 lval = eagle_log1p(val);
        COY_END_PROFILE(ap);
        scalar_vals[i] = lval;
    }
}

#if __AVX2__

static inline void
profile_exp_avx2(f64 *xs, f64 *avx2_vals)
{
    for(size i = 0; i < COUNT_TRIALS; i += 4)
    {
        __m256d xvec = _mm256_loadu_pd(&xs[i]);
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("exp-avx2-test");
        __m256d evec = eagle_avx2_exp_pd(xvec);
        COY_END_PROFILE(ap);
        _mm256_storeu_pd(&avx2_vals[i], evec);
    }
}

static inline void
profile_log_avx2(f64 *xs, f64 *avx2_vals)
{
    for(size i = 0; i < COUNT_TRIALS; i += 4)
    {
        __m256d xvec = _mm256_loadu_pd(&xs[i]);
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log-avx2-test");
        __m256d evec = eagle_avx2_log_pd(xvec);
        COY_END_PROFILE(ap);
        _mm256_storeu_pd(&avx2_vals[i], evec);
    }
}

#endif

#if ELK_AVX_512

static inline void
profile_exp_avx512(f64 *xs, f64 *avx512_vals)
{
    for(size i = 0; i < COUNT_TRIALS; i += 8)
    {
        __m512d xvec = _mm512_loadu_pd(&xs[i]);
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("exp-avx512-test");
        __m512d evec = eagle_avx512_exp_pd(xvec);
        COY_END_PROFILE(ap);
        _mm512_storeu_pd(&avx512_vals[i], evec);
    }
}

static inline void
profile_log_avx512(f64 *xs, f64 *avx512_vals)
{
    for(size i = 0; i < COUNT_TRIALS; i += 8)
    {
        __m512d xvec = _mm512_loadu_pd(&xs[i]);
        CoyProfileAnchor ap = COY_START_PROFILE_BLOCK("log-avx512-test");
        __m512d evec = eagle_avx512_log_pd(xvec);
        COY_END_PROFILE(ap);
        _mm512_storeu_pd(&avx512_vals[i], evec);
    }
}

#endif

i32
main(i32 argc, char *argv[])
{
    coy_profile_begin();

    MagAllocator alloc_ = mag_allocator_dyn_arena_create(1);
    MagAllocator *alloc = &alloc_;

    f64 *xs, *vals;

    allocate_buffers_and_fill_xs(min_exp_arg, max_exp_arg, &xs, &vals, alloc);

    profile_exp_std(xs, vals);
    profile_exp_scalar(xs, vals);

#if __AVX2__
    profile_exp_avx2(xs, vals);
#endif

#if ELK_AVX_512
    profile_exp_avx512(xs, vals);
#endif

    /* Reset for the log tests. */
    eco_arena_reset(alloc);

    allocate_buffers_and_fill_xs(min_log_arg, max_log_arg, &xs, &vals, alloc);

    profile_log_std(xs, vals);
    profile_log_scalar(xs, vals);

#if __AVX2__
    profile_log_avx2(xs, vals);
#endif

#if ELK_AVX_512
    profile_log_avx512(xs, vals);
#endif

    /* Reset for the log tests. */
    eco_arena_reset(alloc);

    allocate_buffers_and_fill_xs(min_log1p_arg, max_log1p_arg, &xs, &vals, alloc);

    profile_log1p_std(xs, vals);
    profile_log1p_scalar(xs, vals);

#if __AVX2__
//    profile_log1p_avx2(xs, vals);
#endif

#if ELK_AVX_512
//    profile_log1p_avx512(xs, vals);
#endif

    coy_profile_end();

#if COY_PROFILE
    printf("Total Runtime = %.3lf seconds at a frequency of %"PRIu64"\n",
            coy_global_profiler.total_elapsed, coy_global_profiler.freq);

    u64 log_std_elapsed = 0;
    u64 log_scalar_elapsed = 0;
    u64 log_avx2_elapsed = 0;
    u64 log_avx512_elapsed = 0;

    u64 exp_std_elapsed = 0;
    u64 exp_scalar_elapsed = 0;
    u64 exp_avx2_elapsed = 0;
    u64 exp_avx512_elapsed = 0;

    u64 log1p_std_elapsed = 0;
    u64 log1p_scalar_elapsed = 0;
    u64 log1p_avx2_elapsed = 0;
    u64 log1p_avx512_elapsed = 0;

    for(i32 i = 0; i < COY_PROFILE_NUM_BLOCKS; ++i)
    {
        CoyBlockProfiler *block = &coy_global_profiler.blocks[i];
        if(block->hit_count)
        {
            ElkStr label = elk_str_from_cstring((char *)block->label);

            if(elk_str_eq(label, (ElkStr){ .start = "log-std-test", .len = 12}))
            {
                log_std_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-std-test", .len = 14}))
            {
                log1p_std_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-std-test", .len = 12}))
            {
                exp_std_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-scalar-test", .len = 15}))
            {
                log_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-scalar-test", .len = 17}))
            {
                log1p_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-scalar-test", .len = 15}))
            {
                exp_scalar_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-avx2-test", .len = 13}))
            {
                log_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-avx2-test", .len = 15}))
            {
                log1p_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-avx2-test", .len = 13}))
            {
                exp_avx2_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log-avx512-test", .len = 15}))
            {
                log_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "log1p-avx512-test", .len = 17}))
            {
                log1p_avx512_elapsed = block->tsc_elapsed_exclusive; 
            }

            if(elk_str_eq(label, (ElkStr){ .start = "exp-avx512-test", .len = 15}))
            {
                exp_avx512_elapsed = block->tsc_elapsed_exclusive; 
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
    printf("  exp() Std     Elapsed = %13ld\n", exp_std_elapsed);
    printf("  exp() Scalar  Elapsed = %13ld\n", exp_scalar_elapsed);
    printf("  exp() AVX2    Elapsed = %13ld\n", exp_avx2_elapsed);
    printf("  exp() AVX5122 Elapsed = %13ld\n", exp_avx512_elapsed);
    printf("  log() Std     Elapsed = %13ld\n", log_std_elapsed);
    printf("  log() Scalar  Elapsed = %13ld\n", log_scalar_elapsed);
    printf("  log() AVX2    Elapsed = %13ld\n", log_avx2_elapsed);
    printf("  log() AVX5122 Elapsed = %13ld\n", log_avx512_elapsed);
    printf("log1p() Std     Elapsed = %13ld\n", log1p_std_elapsed);
    printf("log1p() Scalar  Elapsed = %13ld\n", log1p_scalar_elapsed);
    printf("log1p() AVX2    Elapsed = %13ld\n", log1p_avx2_elapsed);
    printf("log1p() AVX5122 Elapsed = %13ld\n", log1p_avx512_elapsed);
    printf("\n");
    printf("  exp() Scalar Speed up = %.2lf\n", (f64)exp_std_elapsed / (f64)exp_scalar_elapsed);
    printf("  exp() AVX2 Speed up   = %.2lf\n", (f64)exp_std_elapsed / (f64)exp_avx2_elapsed);
    printf("  exp() AVX512 Speed up = %.2lf\n", (f64)exp_std_elapsed / (f64)exp_avx512_elapsed);
    printf("  log() Scalar Speed up = %.2lf\n", (f64)log_std_elapsed / (f64)log_scalar_elapsed);
    printf("  log() AVX2 Speed up   = %.2lf\n", (f64)log_std_elapsed / (f64)log_avx2_elapsed);
    printf("  log() AVX512 Speed up = %.2lf\n", (f64)log_std_elapsed / (f64)log_avx512_elapsed);
    printf("log1p() Scalar Speed up = %.2lf\n", (f64)log1p_std_elapsed / (f64)log1p_scalar_elapsed);
    printf("log1p() AVX2 Speed up   = %.2lf\n", (f64)log1p_std_elapsed / (f64)log1p_avx2_elapsed);
    printf("log1p() AVX512 Speed up = %.2lf\n", (f64)log1p_std_elapsed / (f64)log1p_avx512_elapsed);

#endif
    return EXIT_SUCCESS;
}

COY_PROFILE_STATIC_CHECK ;
