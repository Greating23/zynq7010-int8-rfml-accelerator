#include <climits>
#include <cstdlib>
#include <iostream>

#include "../src/amc_accel.hpp"
#include "../include/test_vector.h"

int main() {
    hls::stream<axis_in_t> input_stream;
    hls::stream<axis_out_t> output_stream;

    for (int idx = 0; idx < RFML_FRAME_LEN; ++idx) {
        axis_in_t word;
        word.data = RFML_INPUT[idx];
        word.keep = -1;
        word.strb = -1;
        word.last = (idx == RFML_FRAME_LEN - 1) ? 1 : 0;
        input_stream.write(word);
    }

    amc_accel(input_stream, output_stream);

    int predicted = 0;
    int32_t best = INT32_MIN;
    int errors = 0;
    for (int cls = 0; cls < RFML_NUM_CLASSES; ++cls) {
        axis_out_t word = output_stream.read();
        int32_t value = static_cast<int32_t>(word.data);
        std::cout << "class " << cls << ": " << value
                  << " expected=" << RFML_EXPECTED_LOGITS[cls] << std::endl;
        if (value != RFML_EXPECTED_LOGITS[cls]) {
            ++errors;
        }
        if (value > best) {
            best = value;
            predicted = cls;
        }
    }

    std::cout << "predicted=" << predicted
              << " expected_class=" << RFML_EXPECTED_CLASS << std::endl;

    if (predicted != RFML_EXPECTED_CLASS) {
        ++errors;
    }
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
