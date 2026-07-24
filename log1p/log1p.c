#include "../tests/eagle_test.h"

static inline f64
eagle_log1p(f64 x)
{

    if(eagle_isnan(x) | (x < -1.0)) { return EAGLE_NAN; }
    if(x == -1.0) { return EAGLE_NEG_INF; }
    if(x == EAGLE_POS_INF) { return EAGLE_POS_INF; }

    f64 const small_x_range_low = -0x1.2bec333018867p-2;
    f64 const small_x_range_high = 0x1.a827999fcef32p-2;

    if((x >= small_x_range_low) & (x <= small_x_range_high))
    {

        f64 const c00 = -0x1p-1;
        f64 const c01 = 0x1.5555555555448p-2;
        f64 const c02 = -0x1.fffffffffff6ap-3;
        f64 const c03 = 0x1.99999999daa57p-3;
        f64 const c04 = -0x1.5555555599db4p-3;
        f64 const c05 = 0x1.249248fff584cp-3;
        f64 const c06 = -0x1.ffffff850d022p-4;
        f64 const c07 = 0x1.c71c8429fd8c4p-4;
        f64 const c08 = -0x1.9999c33c8e46p-4;
        f64 const c09 = 0x1.745ab0d6612b7p-4;
        f64 const c10 = -0x1.554e6bb6b3eb8p-4;
        f64 const c11 = 0x1.3b3ee21ed4d7ap-4;
        f64 const c12 = -0x1.2530c725bbb6cp-4;
        f64 const c13 = 0x1.0f956dd6fd232p-4;
        f64 const c14 = -0x1.f091c4c2a360ep-5;
        f64 const c15 = 0x1.e75b79ecc6381p-5;
        f64 const c16 = -0x1.12ce175d6047cp-4;
        f64 const c17 = 0x1.02e7cbcb1bd0cp-4;
        f64 const c18 = -0x1.d9743e4f79c2ap-6;

        f64 sum = c18;
        sum = __builtin_fma(sum, x, c17);
        sum = __builtin_fma(sum, x, c16);
        sum = __builtin_fma(sum, x, c15);
        sum = __builtin_fma(sum, x, c14);
        sum = __builtin_fma(sum, x, c13);
        sum = __builtin_fma(sum, x, c12);
        sum = __builtin_fma(sum, x, c11);
        sum = __builtin_fma(sum, x, c10);
        sum = __builtin_fma(sum, x, c09);
        sum = __builtin_fma(sum, x, c08);
        sum = __builtin_fma(sum, x, c07);
        sum = __builtin_fma(sum, x, c06);
        sum = __builtin_fma(sum, x, c05);
        sum = __builtin_fma(sum, x, c04);
        sum = __builtin_fma(sum, x, c03);
        sum = __builtin_fma(sum, x, c02);
        sum = __builtin_fma(sum, x, c01);
        sum = __builtin_fma(sum, x, c00);

        return x + x * x * sum; 
    }
    else
    {
        volatile f64 u = 1.0 + x;
        volatile f64 d = u - 1.0;
    
        // To avoid NaN in SIMD when d == 0, we can force d to 1.0. 
        // It doesn't matter what garbage it computes because the blend mask will discard it.
        f64 safe_d = (d == 0.0) ? 1.0 : d; 
        return eagle_log(u) * (x / safe_d);
    }
}

#ifdef __AVX2__

#endif

#if ELK_AVX_512

#endif 

