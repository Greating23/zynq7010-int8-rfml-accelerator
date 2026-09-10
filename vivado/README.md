# Vivado ZYNQ7010 Hardware Platform

## Overview

Vivado hardware project for the ZYNQ7010 INT8 CNN accelerator system.

Integrated IP:

-   ZYNQ Processing System
-   AXI DMA
-   HLS CNN Accelerator IP
-   AXI Interconnect

## Hardware Architecture

    Cortex-A9 PS
          |
          |
       AXI DMA
          |
          |
     AXI4-Stream
          |
          |
    INT8 CNN Accelerator
          |
          |
     AXI DMA
          |
          |
         DDR

## Data Flow

Input:

    ARM Memory
     -> AXI DMA MM2S
     -> CNN Accelerator

Output:

    CNN Accelerator
     -> AXI DMA S2MM
     -> ARM Memory

## Interface

CNN accelerator:

-   s_axis: AXI4-Stream input
-   m_axis: AXI4-Stream output

## Software Data Format

Input:

    uint16_t

    bit[7:0]   : I channel
    bit[15:8]  : Q channel

Output:

    6 INT32 logits

## Build Environment

Tool:

    Vivado 2018.3

Device:

    xc7z010clg400-1

## Validation

Flow:

    Generate Bitstream
            |
    Export Hardware
            |
    SDK Bare-metal Application
            |
    AXI DMA Test
            |
    FPGA CNN Inference

## Performance

FPGA:

    Latency: 1545 us/frame
    Throughput: 647.23 frames/s

ARM Cortex-A9 INT8:

    Latency: 13524 us/frame
    Throughput: 73.94 frames/s

Speedup:

    8.75x
