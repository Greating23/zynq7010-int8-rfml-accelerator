#ifndef RFML_ARM_INT8_CNN_H
#define RFML_ARM_INT8_CNN_H

#include <stdint.h>

#include "weights_arm.h"

/*
 * Input format:
 *
 * packed_input[n]:
 *   bits [7:0]  = I int8
 *   bits [15:8] = Q int8
 *
 * Output:
 *   logits[NUM_CLASSES]
 */
void arm_int8_cnn_infer(
    const uint16_t packed_input[FRAME_LEN],
    int32_t logits[NUM_CLASSES]
);

#endif