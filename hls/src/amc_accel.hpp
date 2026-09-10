#ifndef AMC_ACCEL_HPP
#define AMC_ACCEL_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_axiu<16, 0, 0, 0> axis_in_t;
typedef ap_axiu<32, 0, 0, 0> axis_out_t;

void amc_accel(
    hls::stream<axis_in_t> &s_axis,
    hls::stream<axis_out_t> &m_axis
);

#endif
