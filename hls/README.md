# Vivado HLS INT8 CNN Accelerator

## Overview

This directory contains the Vivado HLS implementation of an INT8 CNN
accelerator deployed on the ZYNQ7010 PL side.

The accelerator communicates with the ARM Cortex-A9 PS through AXI DMA
and AXI4-Stream interfaces.

## Hardware Flow

    Input IQ Samples
            |
            v
    AXI DMA MM2S
            |
            v
    AXI4-Stream
            |
            v
    INT8 CNN Accelerator
            |
            v
    AXI DMA S2MM
            |
            v
    Cortex-A9

## CNN Structure

    Input
    2 x 128 IQ samples

            |
            v

    Conv1D
    Channels: 2 -> 16
    Kernel: 7

            |
            v

    ReLU

            |
            v

    MaxPool

            |
            v

    Conv1D
    Channels: 16 -> 32
    Kernel: 5

            |
            v

    ReLU

            |
            v

    Global Average Pool

            |
            v

    Fully Connected

            |
            v

    6-class logits

## INT8 Quantization

Quantization method:

-   Post Training Quantization (PTQ)
-   Weight: INT8
-   Activation: INT8
-   Bias: INT32
-   Accumulator: INT32

Flow:

    FP32 PyTorch Model
            |
            v
    Calibration
            |
            v
    INT8 Parameters
            |
            v
    weights.h
            |
            v
    HLS Accelerator

## HLS Optimization

Optimization methods:

-   Pipeline with II=1
-   Kernel loop unrolling
-   Partial sum parallel accumulation
-   Memory partitioning

Example:

``` cpp
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
```

## Verification

Verification flow:

    C Simulation
          |
          v
    C/RTL Co-Simulation
          |
          v
    FPGA Hardware Test

Results:

    C Simulation              PASS
    C/RTL Co-Simulation       PASS
    FPGA Hardware             PASS

## Bit Exact Regression

Test:

-   100 input vectors
-   600 output logits

Result:

    Exact logits      : 600 / 600
    Bit-exact frames  : 100 / 100
    Class agreement  : 100 / 100

    FPGA BIT-EXACT REGRESSION PASS

## Device and Tool

Device:

    xc7z010clg400-1

Clock:

    100 MHz

Tool:

    Vivado HLS 2018.3
