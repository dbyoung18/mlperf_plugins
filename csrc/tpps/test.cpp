#include <immintrin.h>
#include <iostream>

static inline __m512h _mm512_tanh_ph(__m512h x){
    auto c0        = _mm512_set1_ph(495.0f);
    auto c1        = _mm512_set1_ph(60.0f);
    auto c1_d      = _mm512_set1_ph(225.0f);
    auto c2_d      = _mm512_set1_ph(10.0f);
    auto c3_d      = _mm512_set1_ph((float)(1.0/21.0)); // 0.0476
    auto hi_bound  = _mm512_set1_ph(4.97f);
    auto lo_bound  = _mm512_set1_ph(-4.97f);
    auto ones      = _mm512_set1_ph(1.0f);
    auto neg_ones  = _mm512_set1_ph(-1.0f);

    auto x2         = _mm512_mul_ph( x, x );
    auto t1_nom     = _mm512_add_ph( x2, c1 );
    auto t2_nom     = _mm512_fmadd_ph( t1_nom, x2, c0);
    auto nom        = _mm512_mul_ph( t2_nom, x);
    auto t1_denom   = _mm512_fmadd_ph( c3_d, x2, c2_d);
    auto t2_denom   = _mm512_fmadd_ph( t1_denom, x2, c1_d);
    auto denom      = _mm512_fmadd_ph( t2_denom, x2, c0 );
    auto denom_rcp  = _mm512_rcp_ph( denom );
    auto mask_hi    = _mm512_cmp_ph_mask( x, hi_bound, _CMP_GT_OQ);
    auto mask_lo    = _mm512_cmp_ph_mask( x, lo_bound, _CMP_LT_OQ);
    auto result     = _mm512_mul_ph( nom, denom_rcp );
    result          = _mm512_mask_blend_ph(mask_hi, result, ones);
    result          = _mm512_mask_blend_ph(mask_lo, result, neg_ones);
    return result;
}

static inline __m512h _mm512_sigmoid_ph(__m512h x){
    auto c0        = _mm512_set1_ph(495.0f);
    auto c1        = _mm512_set1_ph(60.0f);
    auto c1_d      = _mm512_set1_ph(225.0f);
    auto c2_d      = _mm512_set1_ph(10.0f);
    auto c3_d      = _mm512_set1_ph((float)(1.0/21.0)); // 0.0476
    auto hi_bound  = _mm512_set1_ph(4.97f);
    auto lo_bound  = _mm512_set1_ph(-4.97f);
    auto ones      = _mm512_set1_ph(1.0f);
    auto neg_ones  = _mm512_set1_ph(-1.0f);
    auto ph_half   = _mm512_set1_ph(0.5f);

    auto x_half     = _mm512_mul_ph(ph_half,x);
    auto x2         = _mm512_mul_ph( x_half, x_half );
    auto t1_nom     = _mm512_add_ph( x2, c1 );
    auto t2_nom     = _mm512_fmadd_ph( t1_nom, x2, c0);
    auto nom        = _mm512_mul_ph( t2_nom, x_half);
    auto t1_denom   = _mm512_fmadd_ph( c3_d, x2, c2_d);
    auto t2_denom   = _mm512_fmadd_ph( t1_denom, x2, c1_d);
    auto denom      = _mm512_fmadd_ph( t2_denom, x2, c0 );
    auto denom_rcp  = _mm512_rcp_ph( denom );
    auto mask_hi    = _mm512_cmp_ph_mask( x_half, hi_bound, _CMP_GT_OQ);
    auto mask_lo    = _mm512_cmp_ph_mask( x_half, lo_bound, _CMP_LT_OQ);
    auto result     = _mm512_mul_ph( nom, denom_rcp );
    result          = _mm512_mask_blend_ph(mask_hi, result, ones);
    result          = _mm512_mask_blend_ph(mask_lo, result, neg_ones);
    result          = _mm512_add_ph(result,ones);
    result          = _mm512_mul_ph(result,ph_half);
    return result;
}

int main(){
    alignas(64) float x[32];
    auto y = _mm512_set1_ph(1);
    auto z = _mm512_sigmoid_ph(y);
    for(int i=0;i<32;i++){
        x[i] = z[i];
        std::cout << x[i] << std::endl;
    }
    // auto xx = _mm512_cvtph_ph(z);

}

