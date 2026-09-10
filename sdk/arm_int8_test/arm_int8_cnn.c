#include "arm_int8_cnn.h"

#include <stdint.h>

/*
 * Intermediate feature maps.
 *
 * Keep them static instead of putting several KB on the ARM stack.
 */
static int8_t input_buffer[INPUT_CH][FRAME_LEN];

static int8_t conv1_buffer[C1][FRAME_LEN];

static int8_t pool1_buffer[C1][FRAME_LEN / 2];

static int8_t conv2_buffer[C2][FRAME_LEN / 2];

static int8_t gap_buffer[C2];


/* ------------------------------------------------------------
 * Convert packed unsigned byte back to signed INT8.
 * ------------------------------------------------------------ */
static int8_t byte_to_int8(uint8_t value)
{
    if (value < 128U) {
        return (int8_t)value;
    }

    return (int8_t)((int16_t)value - 256);
}


/* ------------------------------------------------------------
 * Same requantization rule as FPGA/HLS:
 *
 * product = acc * multiplier
 *
 * positive:
 *     product += 2^(shift-1)
 *
 * negative:
 *     product -= 2^(shift-1)
 *
 * then arithmetic right shift, ReLU, INT8 saturation.
 *
 * Since every negative result is immediately removed by ReLU,
 * we can directly return zero for negative product.
 * ------------------------------------------------------------ */
static int8_t requant_relu_arm(
    int32_t acc,
    int32_t multiplier
)
{
    int64_t product =
        (int64_t)acc *
        (int64_t)multiplier;

    const int64_t rounding =
        ((int64_t)1 << (REQUANT_SHIFT - 1));

    /*
     * HLS version subtracts rounding, shifts, and then ReLU.
     * Any negative product remains negative, so result is zero.
     */
    if (product < 0) {
        return 0;
    }

    product += rounding;

    int64_t shifted =
        product >> REQUANT_SHIFT;

    if (shifted > 127) {
        return 127;
    }

    return (int8_t)shifted;
}


/* ------------------------------------------------------------
 * Complete INT8 CNN inference
 * ------------------------------------------------------------ */
void arm_int8_cnn_infer(
    const uint16_t packed_input[FRAME_LEN],
    int32_t logits[NUM_CLASSES]
)
{
    int oc;
    int ic;
    int n;
    int k;
    int c;
    int cls;


    /* ========================================================
     * 1. Unpack IQ input
     *
     * packed:
     *   low byte  = I
     *   high byte = Q
     * ======================================================== */
    for (n = 0; n < FRAME_LEN; ++n) {

        uint8_t i_byte =
            (uint8_t)(
                packed_input[n] & 0x00FFU
            );

        uint8_t q_byte =
            (uint8_t)(
                (packed_input[n] >> 8) & 0x00FFU
            );

        input_buffer[0][n] =
            byte_to_int8(i_byte);

        input_buffer[1][n] =
            byte_to_int8(q_byte);
    }


    /* ========================================================
     * 2. Conv1
     *
     * Input:
     *   [2][128]
     *
     * Output:
     *   [16][128]
     *
     * Kernel:
     *   K = 7
     *
     * Padding:
     *   same
     * ======================================================== */
    for (oc = 0; oc < C1; ++oc) {

        for (n = 0; n < FRAME_LEN; ++n) {

            int32_t acc = B1[oc];

            for (ic = 0; ic < INPUT_CH; ++ic) {

                for (k = 0; k < K1; ++k) {

                    int idx =
                        n + k - K1 / 2;

                    if (
                        idx >= 0 &&
                        idx < FRAME_LEN
                    ) {

                        acc +=
                            (int32_t)input_buffer[ic][idx] *
                            (int32_t)W1[oc][ic][k];
                    }
                }
            }

            conv1_buffer[oc][n] =
                requant_relu_arm(
                    acc,
                    M1_INT
                );
        }
    }


    /* ========================================================
     * 3. MaxPool1D
     *
     * [16][128]
     *      ↓
     * [16][64]
     * ======================================================== */
    for (c = 0; c < C1; ++c) {

        for (
            n = 0;
            n < FRAME_LEN / 2;
            ++n
        ) {

            int8_t a =
                conv1_buffer[c][2 * n];

            int8_t b =
                conv1_buffer[c][2 * n + 1];

            pool1_buffer[c][n] =
                (a > b) ? a : b;
        }
    }


    /* ========================================================
     * 4. Conv2
     *
     * Input:
     *   [16][64]
     *
     * Output:
     *   [32][64]
     *
     * Kernel:
     *   K = 5
     * ======================================================== */
    for (oc = 0; oc < C2; ++oc) {

        for (
            n = 0;
            n < FRAME_LEN / 2;
            ++n
        ) {

            int32_t acc = B2[oc];

            for (ic = 0; ic < C1; ++ic) {

                for (k = 0; k < K2; ++k) {

                    int idx =
                        n + k - K2 / 2;

                    if (
                        idx >= 0 &&
                        idx < FRAME_LEN / 2
                    ) {

                        acc +=
                            (int32_t)pool1_buffer[ic][idx] *
                            (int32_t)W2[oc][ic][k];
                    }
                }
            }

            conv2_buffer[oc][n] =
                requant_relu_arm(
                    acc,
                    M2_INT
                );
        }
    }


    /* ========================================================
     * 5. Global Average Pool
     *
     * Exactly follow FPGA/HLS:
     *
     * sum += FRAME_LEN/4
     * gap = sum / (FRAME_LEN/2)
     *
     * FRAME_LEN = 128
     *
     * therefore:
     *     sum += 32
     *     divide by 64
     * ======================================================== */
    for (c = 0; c < C2; ++c) {

        int32_t sum = 0;

        for (
            n = 0;
            n < FRAME_LEN / 2;
            ++n
        ) {

            sum +=
                (int32_t)conv2_buffer[c][n];
        }

        sum += FRAME_LEN / 4;

        gap_buffer[c] =
            (int8_t)(
                sum / (FRAME_LEN / 2)
            );
    }


    /* ========================================================
     * 6. Fully Connected
     *
     * 32 -> 6
     *
     * No final requantization.
     * Output remains INT32 logits.
     * ======================================================== */
    for (
        cls = 0;
        cls < NUM_CLASSES;
        ++cls
    ) {

        int32_t acc =
            B3[cls];

        for (c = 0; c < C2; ++c) {

            acc +=
                (int32_t)gap_buffer[c] *
                (int32_t)W3[cls][c];
        }

        logits[cls] = acc;
    }
}