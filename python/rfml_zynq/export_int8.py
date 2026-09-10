from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import dataset_from_config
from .integer_ref import QuantizedParameters, integer_forward
from .model import model_from_config
from .utils import ensure_dir, load_config, resolve_device, set_seed


def _percentile(values: list[np.ndarray], percentile: float) -> float:
    if not values:
        raise ValueError("No calibration values collected")
    flat = np.concatenate([x.reshape(-1) for x in values])
    value = float(np.percentile(np.abs(flat), percentile))
    return max(value, 1e-8)


@torch.no_grad()
def calibrate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    percentile: float,
) -> tuple[float, float]:
    pool_values: list[np.ndarray] = []
    act2_values: list[np.ndarray] = []

    model.eval()
    for batch_idx, (inputs, _, _) in enumerate(loader):
        inputs = inputs.to(device)
        act1 = model.relu1(model.conv1(inputs))
        pooled = model.pool1(act1)
        act2 = model.relu2(model.conv2(pooled))

        pool_values.append(act1.detach().cpu().numpy())
        act2_values.append(act2.detach().cpu().numpy())
        if batch_idx + 1 >= batches:
            break

    return _percentile(pool_values, percentile), _percentile(act2_values, percentile)


def _quantize_weight(weight: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(weight)))
    scale = max(max_abs / 127.0, 1e-12)
    quantized = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    return quantized, scale


def _quantize_bias(bias: np.ndarray, input_scale: float, weight_scale: float) -> np.ndarray:
    scale = input_scale * weight_scale
    return np.rint(bias / scale).astype(np.int32)


def _c_array(array: np.ndarray, indent: int = 0) -> str:
    if array.ndim == 0:
        return str(array.item())
    pad = " " * indent
    if array.ndim == 1:
        return "{" + ", ".join(str(int(x)) for x in array.tolist()) + "}"
    children = [_c_array(child, indent + 2) for child in array]
    separator = ",\n" + " " * (indent + 2)
    return "{\n" + " " * (indent + 2) + separator.join(children) + "\n" + pad + "}"


def write_weights_header(
    path: Path,
    params: QuantizedParameters,
    classes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_names = ",\n".join(f'    "{name}"' for name in classes)
    content = f"""#ifndef RFML_WEIGHTS_H
#define RFML_WEIGHTS_H

#include <ap_int.h>

static const int FRAME_LEN = 128;
static const int INPUT_CH = 2;
static const int C1 = {params.w1.shape[0]};
static const int K1 = {params.w1.shape[2]};
static const int C2 = {params.w2.shape[0]};
static const int K2 = {params.w2.shape[2]};
static const int NUM_CLASSES = {params.w3.shape[0]};
static const int REQUANT_SHIFT = {params.requant_shift};
static const ap_int<32> M1_INT = {params.m1_int};
static const ap_int<32> M2_INT = {params.m2_int};

static const char *CLASS_NAMES[NUM_CLASSES] = {{
{class_names}
}};

static const ap_int<8> W1[C1][INPUT_CH][K1] = {_c_array(params.w1)};
static const ap_int<32> B1[C1] = {_c_array(params.b1)};
static const ap_int<8> W2[C2][C1][K2] = {_c_array(params.w2)};
static const ap_int<32> B2[C2] = {_c_array(params.b2)};
static const ap_int<8> W3[NUM_CLASSES][C2] = {_c_array(params.w3)};
static const ap_int<32> B3[NUM_CLASSES] = {_c_array(params.b3)};

#endif
"""
    path.write_text(content, encoding="utf-8")

def write_arm_weights_header(
    path: Path,
    params: QuantizedParameters,
    classes: list[str],
) -> None:
    """
    Export quantized CNN parameters for Cortex-A9 C implementation.

    Unlike HLS weights.h, this file uses standard C integer types:
        ap_int<8>  -> int8_t
        ap_int<32> -> int32_t
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    class_names = ",\n".join(f'    "{name}"' for name in classes)
    content = f"""#ifndef RFML_WEIGHTS_ARM_H
#define RFML_WEIGHTS_ARM_H

#include <stdint.h>

#define FRAME_LEN       128
#define INPUT_CH        2

#define C1              {params.w1.shape[0]}
#define K1              {params.w1.shape[2]}

#define C2              {params.w2.shape[0]}
#define K2              {params.w2.shape[2]}

#define NUM_CLASSES     {params.w3.shape[0]}

#define REQUANT_SHIFT   {params.requant_shift}

#define M1_INT          ((int32_t){params.m1_int})
#define M2_INT          ((int32_t){params.m2_int})


static const char * const CLASS_NAMES[NUM_CLASSES] = {{
{class_names}
}};


/* ------------------------------------------------------------
 * Conv1
 *
 * W1:
 *   [C1][INPUT_CH][K1]
 *
 * B1:
 *   [C1]
 * ------------------------------------------------------------ */
static const int8_t W1[C1][INPUT_CH][K1] =
{_c_array(params.w1)};

static const int32_t B1[C1] =
{_c_array(params.b1)};


/* ------------------------------------------------------------
 * Conv2
 *
 * W2:
 *   [C2][C1][K2]
 *
 * B2:
 *   [C2]
 * ------------------------------------------------------------ */
static const int8_t W2[C2][C1][K2] =
{_c_array(params.w2)};

static const int32_t B2[C2] =
{_c_array(params.b2)};


/* ------------------------------------------------------------
 * Fully connected
 *
 * W3:
 *   [NUM_CLASSES][C2]
 *
 * B3:
 *   [NUM_CLASSES]
 * ------------------------------------------------------------ */
static const int8_t W3[NUM_CLASSES][C2] =
{_c_array(params.w3)};

static const int32_t B3[NUM_CLASSES] =
{_c_array(params.b3)};


#endif
"""

    path.write_text(content,encoding="utf-8")

def write_test_vector(
    path: Path,
    input_q: np.ndarray,
    logits: np.ndarray,
    expected_class: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = []
    for idx in range(input_q.shape[1]):
        i_val = int(input_q[0, idx]) & 0xFF
        q_val = int(input_q[1, idx]) & 0xFF
        packed.append(i_val | (q_val << 8))

    packed_array = np.asarray(packed, dtype=np.uint16)
    content = f"""#ifndef RFML_TEST_VECTOR_H
#define RFML_TEST_VECTOR_H

#include <stdint.h>

#define RFML_FRAME_LEN 128
#define RFML_NUM_CLASSES {len(logits)}
#define RFML_EXPECTED_CLASS {expected_class}

static const uint16_t RFML_INPUT[RFML_FRAME_LEN] = {_c_array(packed_array)};
static const int32_t RFML_EXPECTED_LOGITS[RFML_NUM_CLASSES] = {_c_array(logits.astype(np.int32))};

#endif
"""
    path.write_text(content, encoding="utf-8")

def write_multi_test_vectors(
    path: Path,
    inputs_q: np.ndarray,
    logits: np.ndarray,
    expected_classes: np.ndarray,
    true_labels: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    """
    inputs_q:
        [N, 2, 128], int8

    logits:
        [N, NUM_CLASSES], int32

    expected_classes:
        [N]

    true_labels:
        [N]

    sample_indices:
        [N], indices in the original test dataset
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    num_tests = inputs_q.shape[0]
    frame_len = inputs_q.shape[2]
    num_classes = logits.shape[1]

    # FPGA input format:
    # bits [7:0]  = I
    # bits [15:8] = Q
    packed_inputs = np.zeros(
        (num_tests, frame_len),
        dtype=np.uint16,
    )

    for test_idx in range(num_tests):
        for n in range(frame_len):
            i_val = int(inputs_q[test_idx, 0, n]) & 0xFF
            q_val = int(inputs_q[test_idx, 1, n]) & 0xFF

            packed_inputs[test_idx, n] = (
                i_val | (q_val << 8)
            )

    content = f"""#ifndef RFML_MULTI_TEST_VECTORS_H
#define RFML_MULTI_TEST_VECTORS_H

#include <stdint.h>

#define RFML_NUM_TESTS {num_tests}
#define RFML_FRAME_LEN {frame_len}
#define RFML_NUM_CLASSES {num_classes}

/*
 * Packed FPGA input:
 *
 * bits [7:0]  = I int8
 * bits [15:8] = Q int8
 */
static const uint16_t
RFML_INPUTS[RFML_NUM_TESTS][RFML_FRAME_LEN] =
{_c_array(packed_inputs)};

/*
 * Expected integer-reference logits.
 *
 * These values are generated by integer_forward().
 */
static const int32_t
RFML_EXPECTED_LOGITS[RFML_NUM_TESTS][RFML_NUM_CLASSES] =
{_c_array(logits.astype(np.int32))};

/*
 * Expected class from integer reference:
 *
 * argmax(RFML_EXPECTED_LOGITS[test])
 */
static const uint8_t
RFML_EXPECTED_CLASSES[RFML_NUM_TESTS] =
{_c_array(expected_classes.astype(np.uint8))};

/*
 * Ground-truth dataset labels.
 *
 * These are NOT used for bit-exact checking.
 * They are kept only for later accuracy analysis.
 */
static const uint8_t
RFML_TRUE_LABELS[RFML_NUM_TESTS] =
{_c_array(true_labels.astype(np.uint8))};

/*
 * Original indices in the Python test dataset.
 * Useful for reproducing a failed FPGA test.
 */
static const uint32_t
RFML_TEST_INDICES[RFML_NUM_TESTS] =
{_c_array(sample_indices.astype(np.uint32))};

#endif
"""

    path.write_text(content, encoding="utf-8")

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--test-vector", required=True)
    parser.add_argument("--multi-test-vector",default=None,help="Optional output path for multi-sample FPGA regression header.")
    parser.add_argument("--num-test-vectors",type=int,default=100,help="Number of different test samples exported for FPGA regression.")
    parser.add_argument("--arm-header",default=None,help="Optional output path for Cortex-A9 quantized weights header.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))

    model = model_from_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    calibration_ds = dataset_from_config(cfg, "val", seed_offset=10_000_000)
    calibration_loader = DataLoader(
        calibration_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )
    act1_max, act2_max = calibrate(
        model,
        calibration_loader,
        device,
        batches=int(cfg["quant"]["calibration_batches"]),
        percentile=float(cfg["quant"]["percentile"]),
    )

    input_scale = 1.0 / 127.0
    act1_scale = act1_max / 127.0
    act2_scale = act2_max / 127.0

    state = model.state_dict()
    w1, w1_scale = _quantize_weight(state["conv1.weight"].cpu().numpy())
    w2, w2_scale = _quantize_weight(state["conv2.weight"].cpu().numpy())
    w3, w3_scale = _quantize_weight(state["fc.weight"].cpu().numpy())

    b1 = _quantize_bias(state["conv1.bias"].cpu().numpy(), input_scale, w1_scale)
    b2 = _quantize_bias(state["conv2.bias"].cpu().numpy(), act1_scale, w2_scale)
    b3 = _quantize_bias(state["fc.bias"].cpu().numpy(), act2_scale, w3_scale)

    m1 = input_scale * w1_scale / act1_scale
    m2 = act1_scale * w2_scale / act2_scale
    shift = int(cfg["quant"]["requant_shift"])
    m1_int = max(1, int(round(m1 * (1 << shift))))
    m2_int = max(1, int(round(m2 * (1 << shift))))

    params = QuantizedParameters(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
        m1_int=m1_int,
        m2_int=m2_int,
        requant_shift=shift,
    )

    header_path = Path(args.header)
    test_vector_path = Path(args.test_vector)
    write_weights_header(header_path, params, list(cfg["data"]["classes"]))
    # ---------------------------------------------------------
    # Cortex-A9 C weights header
    # ---------------------------------------------------------
    if args.arm_header is not None:
        arm_header_path = Path(args.arm_header)

        write_arm_weights_header(arm_header_path,params,list(cfg["data"]["classes"]))

        print(
            f"Generated ARM weights header: "
            f"{arm_header_path}"
        )

    test_ds = dataset_from_config(cfg, "test", seed_offset=20_000_000)
    sample, label, snr = test_ds[0]
    input_q = np.clip(np.rint(sample.numpy() * 127.0), -127, 127).astype(np.int8)
    logits_int = integer_forward(input_q, params)
    pred_int = int(np.argmax(logits_int))
    write_test_vector(test_vector_path, input_q, logits_int, pred_int)
    # ---------------------------------------------------------
    # Export multiple different test vectors for FPGA regression
    # ---------------------------------------------------------
    if args.multi_test_vector is not None:

        num_tests = min(
            int(args.num_test_vectors),
            len(test_ds),
        )

        if num_tests <= 0:
            raise ValueError(
                "--num-test-vectors must be greater than zero"
            )

        # Spread samples across the entire test dataset instead of
        # simply taking test_ds[0:100].
        #
        # This usually gives better diversity in class / SNR while
        # remaining deterministic and fully reproducible.
        sample_indices = np.linspace(
            0,
            len(test_ds) - 1,
            num=num_tests,
            dtype=np.int64,
        )

        multi_inputs_q = np.zeros(
            (num_tests, 2, 128),
            dtype=np.int8,
        )

        multi_logits = np.zeros(
            (num_tests, params.w3.shape[0]),
            dtype=np.int32,
        )

        multi_expected_classes = np.zeros(
            num_tests,
            dtype=np.uint8,
        )

        multi_true_labels = np.zeros(
            num_tests,
            dtype=np.uint8,
        )

        for test_idx, dataset_idx in enumerate(sample_indices):

            sample_i, label_i, _ = test_ds[int(dataset_idx)]

            input_q_i = np.clip(
                np.rint(sample_i.numpy() * 127.0),
                -127,
                127,
            ).astype(np.int8)

            logits_i = integer_forward(
                input_q_i,
                params,
            ).astype(np.int32)

            pred_i = int(np.argmax(logits_i))

            multi_inputs_q[test_idx] = input_q_i
            multi_logits[test_idx] = logits_i
            multi_expected_classes[test_idx] = pred_i
            multi_true_labels[test_idx] = int(label_i)

        multi_path = Path(args.multi_test_vector)

        write_multi_test_vectors(
            multi_path,
            multi_inputs_q,
            multi_logits,
            multi_expected_classes,
            multi_true_labels,
            sample_indices,
        )

        print(
            f"Generated multi-vector regression header: "
            f"{multi_path}"
        )

        print(
            f"Number of different FPGA test vectors: "
            f"{num_tests}"
        )

        print(
            "Dataset indices: "
            f"{sample_indices.tolist()}"
        )

    matches = 0
    count = min(256, len(test_ds))
    for idx in range(count):
        sample, _, _ = test_ds[idx]
        float_pred = int(model(sample.unsqueeze(0).to(device)).argmax(dim=1).item())
        sample_q = np.clip(np.rint(sample.numpy() * 127.0), -127, 127).astype(np.int8)
        int_pred = int(np.argmax(integer_forward(sample_q, params)))
        matches += int(float_pred == int_pred)

    output_dir = ensure_dir(Path(cfg["output_dir"]) / "quant")
    np.savez(
        output_dir / "int8_parameters.npz",
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
        m1_int=m1_int,
        m2_int=m2_int,
        requant_shift=shift,
        input_scale=input_scale,
        act1_scale=act1_scale,
        act2_scale=act2_scale,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w3_scale=w3_scale,
    )

    print(f"Calibration act1 max: {act1_max:.6f}")
    print(f"Calibration act2 max: {act2_max:.6f}")
    print(f"Float/int class agreement on {count} samples: {matches / count:.4f}")
    print(f"Regression sample label={label}, snr={float(snr):.2f} dB")
    print(f"Integer prediction={pred_int}, logits={logits_int.tolist()}")
    print(f"Generated: {header_path}")
    print(f"Generated: {test_vector_path}")


if __name__ == "__main__":
    main()
