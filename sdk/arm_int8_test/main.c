#include <stdint.h>

#include "xil_printf.h"
#include "xstatus.h"
#include "xtime_l.h"

#include "arm_int8_cnn.h"
#include "test_vector.h"

#define WARMUP_FRAMES  10
#define TEST_FRAMES    1000


static int argmax(
    const int32_t *values,
    int length
)
{
    int best_index = 0;
    int32_t best_value = values[0];

    int idx;

    for (idx = 1; idx < length; ++idx) {

        if (values[idx] > best_value) {
            best_value = values[idx];
            best_index = idx;
        }
    }

    return best_index;
}


/* ------------------------------------------------------------
 * Print XTime ticks as microseconds.
 *
 * Example:
 *   12345.678 us
 * ------------------------------------------------------------ */
static void print_ticks_as_us(
    const char *name,
    XTime ticks
)
{
    uint64_t ns =
        ((uint64_t)ticks * 1000000000ULL) /
        (uint64_t)COUNTS_PER_SECOND;

    unsigned long us_integer =
        (unsigned long)(ns / 1000ULL);

    unsigned long us_fraction =
        (unsigned long)(ns % 1000ULL);

    xil_printf(
        "%s%lu.%03lu us\r\n",
        name,
        us_integer,
        us_fraction
    );
}


int main(void)
{
    static int32_t logits[NUM_CLASSES];

    int frame;
    int cls;

    xil_printf("\r\n");
    xil_printf(
        "========================================\r\n"
    );
    xil_printf(
        " Cortex-A9 INT8 CNN Performance Test\r\n"
    );
    xil_printf(
        "========================================\r\n"
    );


    /* ========================================================
     * 1. Warm-up
     *
     * Do not include these runs in statistics.
     * ======================================================== */
    xil_printf(
        "Running %d warm-up frames...\r\n",
        WARMUP_FRAMES
    );

    for (
        frame = 0;
        frame < WARMUP_FRAMES;
        ++frame
    ) {

        arm_int8_cnn_infer(
            RFML_INPUT,
            logits
        );
    }

    xil_printf(
        "Warm-up finished.\r\n"
    );


    /* ========================================================
     * 2. Performance measurement
     * ======================================================== */
    xil_printf(
        "Running %d measured frames...\r\n",
        TEST_FRAMES
    );

    uint64_t total_ticks = 0;

    XTime min_ticks =
        (XTime)(~(XTime)0);

    XTime max_ticks = 0;


    for (
        frame = 0;
        frame < TEST_FRAMES;
        ++frame
    ) {

        XTime start_time;
        XTime end_time;
        XTime elapsed;


        /*
         * Timing starts immediately before CNN inference.
         */
        XTime_GetTime(
            &start_time
        );


        arm_int8_cnn_infer(
            RFML_INPUT,
            logits
        );


        /*
         * Timing ends immediately after CNN inference.
         */
        XTime_GetTime(
            &end_time
        );


        elapsed =
            end_time - start_time;


        total_ticks +=
            (uint64_t)elapsed;


        if (
            elapsed < min_ticks
        ) {
            min_ticks = elapsed;
        }


        if (
            elapsed > max_ticks
        ) {
            max_ticks = elapsed;
        }
    }


    /* ========================================================
     * 3. Performance statistics
     * ======================================================== */
    XTime average_ticks =
        (XTime)(
            total_ticks /
            TEST_FRAMES
        );


    /*
     * FPS * 100
     *
     * Avoid floating point printf.
     */
    uint64_t fps_x100 =
        (
            (uint64_t)COUNTS_PER_SECOND *
            (uint64_t)TEST_FRAMES *
            100ULL
        )
        /
        total_ticks;


    /* ========================================================
     * 4. Verify that the final output is still bit-exact
     * ======================================================== */
    int exact_logits = 0;

    xil_printf("\r\n");

    xil_printf(
        "Final ARM logits:\r\n"
    );


    for (
        cls = 0;
        cls < NUM_CLASSES;
        ++cls
    ) {

        xil_printf(
            "  class %d: %ld, expected %ld\r\n",
            cls,
            (long)logits[cls],
            (long)RFML_EXPECTED_LOGITS[cls]
        );


        if (
            logits[cls]
            ==
            RFML_EXPECTED_LOGITS[cls]
        ) {

            exact_logits++;
        }
    }


    int predicted =
        argmax(
            logits,
            NUM_CLASSES
        );


    xil_printf(
        "predicted=%d, expected=%d\r\n",
        predicted,
        RFML_EXPECTED_CLASS
    );


    /* ========================================================
     * 5. Final results
     * ======================================================== */
    xil_printf("\r\n");

    xil_printf(
        "========================================\r\n"
    );

    xil_printf(
        " ARM Performance Results\r\n"
    );

    xil_printf(
        "========================================\r\n"
    );


    xil_printf(
        "Measured frames : %d\r\n",
        TEST_FRAMES
    );


    print_ticks_as_us(
        "Average latency : ",
        average_ticks
    );


    print_ticks_as_us(
        "Minimum latency : ",
        min_ticks
    );


    print_ticks_as_us(
        "Maximum latency : ",
        max_ticks
    );


    xil_printf(
        "Throughput      : %lu.%02lu frames/s\r\n",
        (unsigned long)(
            fps_x100 / 100ULL
        ),
        (unsigned long)(
            fps_x100 % 100ULL
        )
    );


    xil_printf(
        "Exact logits    : %d / %d\r\n",
        exact_logits,
        NUM_CLASSES
    );


    xil_printf(
        "========================================\r\n"
    );


    if (
        exact_logits == NUM_CLASSES
        &&
        predicted == RFML_EXPECTED_CLASS
    ) {

        xil_printf(
            "RESULT: ARM PERFORMANCE TEST PASS\r\n"
        );

        return XST_SUCCESS;
    }


    xil_printf(
        "RESULT: ARM PERFORMANCE TEST FAIL\r\n"
    );

    return XST_FAILURE;
}
