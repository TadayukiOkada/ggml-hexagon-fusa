// Run on the Hexagon ISA simulator; independent scalar dequantization + tile oracle.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_vtcm_mgr.h>
#include <qurt_hvx.h>
#include "hmx-kv-prep.h"

static float read_half(const uint8_t * p) {
    __fp16 v;
    memcpy(&v, p, 2);
    return (float) v;
}

static float reference(const uint8_t * row, unsigned i, int type) {
    static const int lut[] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    if (type == HTP_TYPE_F8_E4M3) return (__fp16) ggml_f8_e4m3_to_fp32(row[i]);
    unsigned size = htp_tensor_get_row_size(type, 32);
    const uint8_t * b = row + (i / 32) * size;
    unsigned j = i % 32;
    unsigned off = type == HTP_TYPE_Q5_0 ? 6 : type == HTP_TYPE_Q5_1 ? 8 : type == HTP_TYPE_Q4_1 ? 4 : 2;
    int q = (b[off + j % 16] >> (j >= 16 ? 4 : 0)) & 15;
    if (type == HTP_TYPE_Q5_0 || type == HTP_TYPE_Q5_1) {
        uint32_t high;
        memcpy(&high, b + off - 4, 4);
        q |= ((high >> j) & 1) << 4;
    }
    if (type == HTP_TYPE_Q4_0) q -= 8;
    if (type == HTP_TYPE_Q5_0) q -= 16;
    if (type == HTP_TYPE_IQ4_NL) q = lut[q];
    float result = read_half(b) * q;
    if (type == HTP_TYPE_Q4_1 || type == HTP_TYPE_Q5_1) result += read_half(b + 2);
    return result == 0 ? (__fp16) 0 : (__fp16) result;
}

static int same(const void * a, const void * b, size_t bytes) {
    const uint16_t * x = a, * y = b;
    for (size_t i = 0; i < bytes / 2; ++i) {
        // NaN payload and the sign of exact zero are immaterial to attention.
        if (x[i] == y[i] || (!(x[i] & 0x7fff) && !(y[i] & 0x7fff)) ||
            ((x[i] & 0x7fff) > 0x7c00 && (y[i] & 0x7fff) > 0x7c00)) continue;
        printf("offset=%u ref=%04x got=%04x\n", (unsigned) i, x[i], y[i]);
        return 0;
    }
    return 1;
}

int main(void) {
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) return 2;
    uint8_t * base = HAP_request_VTCM(2*1024*1024, 1);
    if (!base) return 3;
#ifdef KV_PREP_EXHAUSTIVE
    const unsigned rows_all[] = {1,2,3,31,32,33,63,64,65,127};
    const unsigned dims[] = {32,64,96,128,160,256,576};
#else
    const unsigned rows_all[] = {1,3,33};
    const unsigned dims[] = {32,96,128};
#endif
    const int types[] = {HTP_TYPE_Q4_0,HTP_TYPE_Q4_1,HTP_TYPE_Q5_0,HTP_TYPE_Q5_1,HTP_TYPE_IQ4_NL,HTP_TYPE_F8_E4M3};
    const uint16_t scales[] = {0,0x8000,1,0x8400,0x3000,0xbc00,0x6000};
    unsigned cases = 0;
    for (unsigned ti=0; ti<6; ++ti) for (unsigned packed=0;packed<2;++packed)
    for (unsigned ri=0;ri<sizeof(rows_all)/sizeof(rows_all[0]);++ri) for (unsigned di=0;di<sizeof(dims)/sizeof(dims[0]);++di) {
        if (!packed && !ri && !di) { printf("type=%d\n", types[ti]); fflush(stdout); }
        int type = types[ti];
        unsigned rows=rows_all[ri], dim=dims[di], padded=(dim+63)/64*64;
        unsigned stride=packed ? htp_tensor_get_row_size(type,dim) : padded*2, rt=(rows+31)/32+1;
        size_t bytes=(size_t)rt*(padded/32)*2048;
        uint8_t * src=base, * copy=base+192*1024;
        __fp16 * plain=(void *)(base+384*1024), * ref=(void *)(base+640*1024), * got=(void *)(base+1024*1024);
        memset(src,0,(size_t)rows*stride+128);
        for (unsigned r=0;r<rows;++r) {
            uint8_t * row=src+(size_t)r*stride;
            for (unsigned j=0;j<htp_tensor_get_row_size(type,dim);++j) row[j]=(uint8_t)(r*31+j*17+ri);
            if (type != HTP_TYPE_F8_E4M3) for (unsigned j=0;j<dim/32;++j) {
                uint8_t * b=row+j*htp_tensor_get_row_size(type,32);
                uint16_t sc=scales[(r+j)%7]; memcpy(b,&sc,2);
                if (type==HTP_TYPE_Q4_1 || type==HTP_TYPE_Q5_1) { sc=scales[(r+j+3)%7]; memcpy(b+2,&sc,2); }
            }
            for (unsigned j=0;j<padded;++j) plain[r*padded+j]=j<dim ? reference(row,j,type) : (__fp16)0;
        }
        memcpy(copy,src,(size_t)rows*stride+128);
        for (unsigned threads=1;threads<=4;threads*=2) for (unsigned kind=0;kind<2;++kind) {
            memset(ref,0x55,bytes+128); memset(got,0x55,bytes+128);
            unsigned span=(((rows+threads-1)/threads)+1)/2*2;
            for (unsigned t=0;t<threads;++t) {
                unsigned start=t*span, end=MIN(start+span,rows);
                if (start>=end) continue;
                if (!kind) {
                    hmx_interleave_rows_to_tiles(ref,plain,rows,padded,padded,start,end); hvx_gather_sync(ref);
                    hmx_kv_k_to_tiles(got,src,stride,dim,padded,type,start,end); hvx_gather_sync(got);
                } else {
                    hmx_interleave_cols_to_tiles(ref,plain,rows,padded,padded,rt,start,end);
                    hmx_kv_v_to_tiles(got,src,stride,dim,padded,rt,type,start,end);
                }
            }
            if (!same(ref,got,bytes+128) || memcmp(src,copy,(size_t)rows*stride+128)) {
                printf("FAIL type=%d packed=%u rows=%u dim=%u threads=%u kind=%u\n",type,packed,rows,dim,threads,kind); return 1;
            }
            ++cases;
        }
        // Expansion in-place is used by the HVX fallback; check its guard too.
        for (unsigned r=0;r<rows;++r) {
            memcpy(got,src+(size_t)r*stride,htp_tensor_get_row_size(type,dim));
            memset((uint8_t *)got+dim*2,0x55,128);
            hmx_kv_row_to_f16(got,got,dim,type);
            if (!same(plain+r*padded,got,dim*2)) { printf("FAIL row type=%d dim=%u\n",type,dim); return 1; }
            for (unsigned j=0;j<128;++j) if (((uint8_t *)got)[dim*2+j]!=0x55) return 4;
        }
    }
    HAP_release_VTCM(base); qurt_hvx_unlock();
    printf("PASS %u tile comparisons, packed/padded rows, scalar oracle, input/guards and in-place decode\n",cases);
    return 0;
}
