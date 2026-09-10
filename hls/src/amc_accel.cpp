#include "amc_accel.hpp"
#include "../include/weights.h"

static ap_int<8> requant_relu(ap_int<32> acc, ap_int<32> multiplier) {
#pragma HLS INLINE
    ap_int<64> product = (ap_int<64>)acc * (ap_int<64>)multiplier;
    const ap_int<64> rounding = ((ap_int<64>)1) << (REQUANT_SHIFT - 1);
    if (product >= 0) {
        product += rounding;
    } else {
        product -= rounding;
    }
    ap_int<64> shifted = product >> REQUANT_SHIFT;
    if (shifted < 0) {
        return 0;
    }
    if (shifted > 127) {
        return 127;
    }
    return (ap_int<8>)shifted;
}

void amc_accel(
    hls::stream<axis_in_t> &s_axis,
    hls::stream<axis_out_t> &m_axis
) {
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return

    ap_int<8> input[INPUT_CH][FRAME_LEN];
    ap_int<8> conv1[C1][FRAME_LEN];
    ap_int<8> pool1[C1][FRAME_LEN / 2];
    ap_int<8> conv2[C2][FRAME_LEN / 2];
    ap_int<8> gap[C2];

#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=gap complete dim=1
#pragma HLS ARRAY_PARTITION variable=pool1 cyclic factor=5 dim=2
#pragma HLS ARRAY_PARTITION variable=W2 complete dim=3

read_input:
    for (int n = 0; n < FRAME_LEN; ++n) {
#pragma HLS PIPELINE II=1
        axis_in_t word = s_axis.read();
        input[0][n] = (ap_int<8>)word.data.range(7, 0);
        input[1][n] = (ap_int<8>)word.data.range(15, 8);
    }

conv1_out:
    for (int oc = 0; oc < C1; ++oc) {
        for (int n = 0; n < FRAME_LEN; ++n) {
            ap_int<32> acc = B1[oc];
        conv1_reduce:
            for (int ic = 0; ic < INPUT_CH; ++ic) {
                for (int k = 0; k < K1; ++k) {
#pragma HLS PIPELINE II=1
                    const int idx = n + k - K1 / 2;
                    if (idx >= 0 && idx < FRAME_LEN) {
                        acc += (ap_int<16>)input[ic][idx] * W1[oc][ic][k];
                    }
                }
            }
            conv1[oc][n] = requant_relu(acc, M1_INT);
        }
    }

pool1_out:
    for (int c = 0; c < C1; ++c) {
        for (int n = 0; n < FRAME_LEN / 2; ++n) {
#pragma HLS PIPELINE II=1
            ap_int<8> a = conv1[c][2 * n];
            ap_int<8> b = conv1[c][2 * n + 1];
            pool1[c][n] = (a > b) ? a : b;
        }
    }

conv2_out:
	for (int oc = 0; oc < C2; ++oc) {
		for (int n = 0; n < FRAME_LEN / 2; ++n) {

			// 两套独立的部分和：
			// partial_even 处理 ic = 0,2,4,...
			// partial_odd  处理 ic = 1,3,5,...
			ap_int<32> partial_even[K2];
			ap_int<32> partial_odd[K2];

#pragma HLS ARRAY_PARTITION variable=partial_even complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial_odd  complete dim=1

		conv2_init_partial:
			for (int k = 0; k < K2; ++k) {
#pragma HLS UNROLL
				partial_even[k] = 0;
				partial_odd[k]  = 0;
			}

		// C1=16，所以每次处理两个通道，总共8轮
		conv2_pair:
			for (int p = 0; p < C1 / 2; ++p) {
#pragma HLS PIPELINE II=1

				const int ic_even = 2 * p;
				const int ic_odd  = 2 * p + 1;

			conv2_k_parallel:
				for (int k = 0; k < K2; ++k) {
#pragma HLS UNROLL

					const int idx = n + k - K2 / 2;

					if (idx >= 0 && idx < FRAME_LEN / 2) {

						partial_even[k] +=
							(ap_int<16>)pool1[ic_even][idx]
							* W2[oc][ic_even][k];

						partial_odd[k] +=
							(ap_int<16>)pool1[ic_odd][idx]
							* W2[oc][ic_odd][k];
					}
				}
			}

			// 最后把两套5路部分和归约回来
			ap_int<32> acc = B2[oc];

		conv2_final_reduce:
			for (int k = 0; k < K2; ++k) {
#pragma HLS PIPELINE II=1
				acc += partial_even[k];
				acc += partial_odd[k];
			}

			conv2[oc][n] = requant_relu(acc, M2_INT);
		}
	}

global_average_pool:
    for (int c = 0; c < C2; ++c) {
        ap_int<32> sum = 0;
        for (int n = 0; n < FRAME_LEN / 2; ++n) {
#pragma HLS PIPELINE II=1
            sum += conv2[c][n];
        }
        sum += FRAME_LEN / 4;
        gap[c] = (ap_int<8>)(sum / (FRAME_LEN / 2));
    }

fully_connected:
    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
        ap_int<32> acc = B3[cls];
        for (int c = 0; c < C2; ++c) {
#pragma HLS PIPELINE II=1
            acc += (ap_int<16>)gap[c] * W3[cls][c];
        }
        axis_out_t out_word;
        out_word.data = (ap_uint<32>)acc;
        out_word.keep = -1;
        out_word.strb = -1;
        out_word.last = (cls == NUM_CLASSES - 1) ? 1 : 0;
        m_axis.write(out_word);
    }
}
