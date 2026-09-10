#include <stdint.h>
#include <stdio.h>

#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"

#include "multi_test_vectors.h"

#ifndef XPAR_AXIDMA_0_DEVICE_ID
#error "AXI DMA device ID not found."
#endif

#define DMA_DEVICE_ID XPAR_AXIDMA_0_DEVICE_ID

#define MAX_FAILURES_TO_PRINT 10

static XAxiDma dma_instance;

static uint16_t tx_buffer[RFML_FRAME_LEN]
    __attribute__((aligned(64)));

static int32_t rx_buffer[RFML_NUM_CLASSES]
    __attribute__((aligned(64)));


/* ------------------------------------------------------------
 * DMA initialization
 * ------------------------------------------------------------ */
static int init_dma(void)
{
    XAxiDma_Config *config;
    int status;

    config = XAxiDma_LookupConfig(DMA_DEVICE_ID);

    if (config == NULL) {
        return XST_FAILURE;
    }

    status = XAxiDma_CfgInitialize(
        &dma_instance,
        config
    );

    if (status != XST_SUCCESS) {
        return status;
    }

    if (XAxiDma_HasSg(&dma_instance)) {
        xil_printf(
            "ERROR: AXI DMA must use simple mode.\r\n"
        );
        return XST_FAILURE;
    }

    XAxiDma_IntrDisable(
        &dma_instance,
        XAXIDMA_IRQ_ALL_MASK,
        XAXIDMA_DEVICE_TO_DMA
    );

    XAxiDma_IntrDisable(
        &dma_instance,
        XAXIDMA_IRQ_ALL_MASK,
        XAXIDMA_DMA_TO_DEVICE
    );

    return XST_SUCCESS;
}


/* ------------------------------------------------------------
 * Argmax
 * ------------------------------------------------------------ */
static int argmax(
    const int32_t *values,
    int length
)
{
    int best_index = 0;
    int32_t best_value = values[0];

    for (int idx = 1; idx < length; ++idx) {

        if (values[idx] > best_value) {
            best_value = values[idx];
            best_index = idx;
        }
    }

    return best_index;
}


/* ------------------------------------------------------------
 * Run one FPGA inference
 * ------------------------------------------------------------ */
static int run_fpga_inference(void)
{
    int status;

    Xil_DCacheFlushRange(
        (UINTPTR)tx_buffer,
        sizeof(tx_buffer)
    );

    Xil_DCacheFlushRange(
        (UINTPTR)rx_buffer,
        sizeof(rx_buffer)
    );

    /*
     * Start S2MM first.
     */
    status = XAxiDma_SimpleTransfer(
        &dma_instance,
        (UINTPTR)rx_buffer,
        sizeof(rx_buffer),
        XAXIDMA_DEVICE_TO_DMA
    );

    if (status != XST_SUCCESS) {
        return XST_FAILURE;
    }

    /*
     * Then start MM2S.
     */
    status = XAxiDma_SimpleTransfer(
        &dma_instance,
        (UINTPTR)tx_buffer,
        sizeof(tx_buffer),
        XAXIDMA_DMA_TO_DEVICE
    );

    if (status != XST_SUCCESS) {
        return XST_FAILURE;
    }

    while (
        XAxiDma_Busy(
            &dma_instance,
            XAXIDMA_DMA_TO_DEVICE
        )
    ) {
    }

    while (
        XAxiDma_Busy(
            &dma_instance,
            XAXIDMA_DEVICE_TO_DMA
        )
    ) {
    }

    Xil_DCacheInvalidateRange(
        (UINTPTR)rx_buffer,
        sizeof(rx_buffer)
    );

    return XST_SUCCESS;
}


int main(void)
{
    int exact_frames = 0;
    int exact_logits = 0;
    int class_matches = 0;

    int failed_frames = 0;
    int printed_failures = 0;

    const int total_logits =
        RFML_NUM_TESTS * RFML_NUM_CLASSES;

    xil_printf("\r\n");
    xil_printf(
        "============================================\r\n"
    );
    xil_printf(
        " RFML ZYNQ7010 Multi-Vector Regression Test\r\n"
    );
    xil_printf(
        "============================================\r\n"
    );

    xil_printf(
        "Different test vectors: %d\r\n",
        RFML_NUM_TESTS
    );

    if (init_dma() != XST_SUCCESS) {

        xil_printf(
            "ERROR: DMA initialization failed.\r\n"
        );

        return XST_FAILURE;
    }

    xil_printf("DMA initialization OK.\r\n");
    xil_printf("Starting FPGA regression...\r\n\r\n");


    /* --------------------------------------------------------
     * Test all different vectors
     * -------------------------------------------------------- */
    for (
        int test = 0;
        test < RFML_NUM_TESTS;
        ++test
    ) {

        /*
         * Copy this test vector into DMA TX buffer.
         */
        for (
            int n = 0;
            n < RFML_FRAME_LEN;
            ++n
        ) {
            tx_buffer[n] =
                RFML_INPUTS[test][n];
        }

        /*
         * Clear old result.
         */
        for (
            int cls = 0;
            cls < RFML_NUM_CLASSES;
            ++cls
        ) {
            rx_buffer[cls] = 0;
        }


        /* ----------------------------------------------------
         * FPGA inference
         * ---------------------------------------------------- */
        if (
            run_fpga_inference()
            != XST_SUCCESS
        ) {

            xil_printf(
                "ERROR: DMA failed at test %d "
                "(dataset index %lu).\r\n",
                test,
                (unsigned long)
                RFML_TEST_INDICES[test]
            );

            return XST_FAILURE;
        }


        /* ----------------------------------------------------
         * Compare all six logits
         * ---------------------------------------------------- */
        int frame_exact = 1;

        for (
            int cls = 0;
            cls < RFML_NUM_CLASSES;
            ++cls
        ) {

            if (
                rx_buffer[cls]
                ==
                RFML_EXPECTED_LOGITS[test][cls]
            ) {

                exact_logits++;
            }
            else {

                frame_exact = 0;
            }
        }


        if (frame_exact) {

            exact_frames++;
        }
        else {

            failed_frames++;

            /*
             * Print only first few failures so UART output
             * does not become excessive.
             */
            if (
                printed_failures
                <
                MAX_FAILURES_TO_PRINT
            ) {

                xil_printf(
                    "FAIL test=%d, dataset_index=%lu\r\n",
                    test,
                    (unsigned long)
                    RFML_TEST_INDICES[test]
                );

                for (
                    int cls = 0;
                    cls < RFML_NUM_CLASSES;
                    ++cls
                ) {

                    xil_printf(
                        "  class %d: FPGA=%ld expected=%ld\r\n",
                        cls,
                        (long)rx_buffer[cls],
                        (long)
                        RFML_EXPECTED_LOGITS[test][cls]
                    );
                }

                printed_failures++;
            }
        }


        /* ----------------------------------------------------
         * Class agreement with Python integer reference
         * ---------------------------------------------------- */
        int fpga_class =
            argmax(
                rx_buffer,
                RFML_NUM_CLASSES
            );

        if (
            fpga_class
            ==
            RFML_EXPECTED_CLASSES[test]
        ) {

            class_matches++;
        }
        else {

            if (
                printed_failures
                <
                MAX_FAILURES_TO_PRINT
            ) {

                xil_printf(
                    "CLASS FAIL test=%d: "
                    "FPGA=%d expected=%d\r\n",
                    test,
                    fpga_class,
                    RFML_EXPECTED_CLASSES[test]
                );

                printed_failures++;
            }
        }


        /*
         * Small progress message.
         */
        if (
            ((test + 1) % 10) == 0
        ) {

            xil_printf(
                "Completed %d / %d\r\n",
                test + 1,
                RFML_NUM_TESTS
            );
        }
    }


    /* --------------------------------------------------------
     * Final result
     * -------------------------------------------------------- */
    xil_printf("\r\n");
    xil_printf(
        "============================================\r\n"
    );
    xil_printf(
        " Multi-Vector Regression Results\r\n"
    );
    xil_printf(
        "============================================\r\n"
    );

    xil_printf(
        "Different test vectors : %d\r\n",
        RFML_NUM_TESTS
    );

    xil_printf(
        "Exact logits           : %d / %d\r\n",
        exact_logits,
        total_logits
    );

    xil_printf(
        "Bit-exact frames       : %d / %d\r\n",
        exact_frames,
        RFML_NUM_TESTS
    );

    xil_printf(
        "Class agreement        : %d / %d\r\n",
        class_matches,
        RFML_NUM_TESTS
    );

    xil_printf(
        "Failed frames          : %d\r\n",
        failed_frames
    );

    xil_printf(
        "============================================\r\n"
    );


    /* --------------------------------------------------------
     * Pass / fail
     * -------------------------------------------------------- */
    if (
        exact_frames == RFML_NUM_TESTS
        &&
        exact_logits == total_logits
        &&
        class_matches == RFML_NUM_TESTS
    ) {

        xil_printf(
            "RESULT: FPGA BIT-EXACT REGRESSION PASS\r\n"
        );

        return XST_SUCCESS;
    }


    xil_printf(
        "RESULT: FPGA BIT-EXACT REGRESSION FAIL\r\n"
    );

    return XST_FAILURE;
}
