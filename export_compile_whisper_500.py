#!/usr/bin/env python3
"""
export_compile_whisper_500.py
==============================
Whisper Small (500-mel / ~5s) AIMET w8a16 양자화 + QNN SDK 2.37 로컬 컴파일
타겟: QCS6490 (SM7325), HTP V68, QNN 2.37

사용법
------
  # 전체 파이프라인 (양자화 → 컴파일)
  export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.37.0.240927
  python scripts/export_compile_whisper_500.py \\
      --calib-npz calib_out/calib_whisper_5s_enko.npz \\
      --output-dir calib_out/whisper_small_q_500

  # 양자화만 (컴파일 생략)
  python scripts/export_compile_whisper_500.py \\
      --calib-npz calib_out/calib_whisper_5s_enko.npz \\
      --output-dir calib_out/whisper_small_q_500 \\
      --skip-compile

  # 기존 AIMET 산출물로 컴파일만
  export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.37.0.240927
  python scripts/export_compile_whisper_500.py \\
      --output-dir calib_out/whisper_small_q_500 \\
      --compile-only

  # 인코더만 컴파일
  python scripts/export_compile_whisper_500.py \\
      --output-dir calib_out/whisper_small_q_500 \\
      --compile-only --encoder-only

필수 패키지
-----------
  pip install torch onnx onnxsim numpy
  pip install aimet-onnx            # AIMET 2.20+
  pip install qai-hub-models        # WhisperSmall, WHISPER_AIMET_CONFIG 등

필수 환경 (컴파일 시)
---------------------
  QNN SDK 2.37 설치 후:
    export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.37.0.240927

  aarch64-android 타겟 시 Android NDK + clang cross-toolchain이 PATH에 필요.

산출물
------
  {output-dir}/
    encoder_aimet/
      model.onnx          ← AIMET ONNX (양자화 파라미터 포함)
      model.encodings     ← AIMET 양자화 인코딩 (scale/offset/bw)
    decoder_aimet/
      model.onnx
      model.encodings
    encoder_precompiled/
      model.onnx          ← QNN ONNX 래퍼 (또는 AIMET ONNX 복사본)
      model.bin           ← HTP V68 직렬화 컨텍스트 바이너리
    decoder_precompiled/
      model.onnx
      model.bin
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

import aimet_onnx
from aimet_common.defs import QuantScheme
from aimet_onnx.cross_layer_equalization import equalize_model
from aimet_onnx.quantsim import QuantizationSimModel
from onnxsim import simplify

from qai_hub_models.models._shared.hf_whisper.model import (
    MEAN_DECODE_LEN,
    HfWhisperDecoder,
    HfWhisperEncoder,
)
from qai_hub_models.models._shared.hf_whisper_quantized.model import WHISPER_AIMET_CONFIG
from qai_hub_models.models.whisper_small.model import WhisperSmall
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.onnx.helpers import safe_torch_onnx_export

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
MEL_LEN = 500            # mel time-steps (~5s audio)
AUDIO_EMB_LEN = 250      # MEL_LEN / 2 (conv2 stride=2)
NUM_BLOCKS = 12
ATTENTION_DIM = 768
NUM_HEADS = 12
HEAD_DIM = ATTENTION_DIM // NUM_HEADS  # 64

# QCS6490: SM7325 기반, Hexagon HTP V68
HTP_ARCH_VERSION = 68
SOC_ID_QCS6490 = 1045    # QNN SOC ID for QCS6490

SOT_TOKEN_ID = 50258     # <|startoftranscript|>
MASK_NEG = -100.0

DEFAULT_CALIB_NPZ = "calib_out/calib_whisper_5s_enko.npz"
DEFAULT_OUTPUT_DIR = "calib_out/whisper_small_q_500"


# ---------------------------------------------------------------------------
# 입력 스펙 (ONNX export / calibration용 float32)
# ---------------------------------------------------------------------------

def encoder_input_spec() -> dict:
    return {"input_features": ((1, 80, MEL_LEN), "float32")}


def decoder_input_spec() -> dict:
    specs: dict = {
        "input_ids": ((1, 1), "int32"),
        "attention_mask": ((1, 1, 1, MEAN_DECODE_LEN), "float32"),
    }
    for i in range(NUM_BLOCKS):
        specs[f"k_cache_self_{i}_in"] = ((NUM_HEADS, 1, HEAD_DIM, MEAN_DECODE_LEN - 1), "float32")
        specs[f"v_cache_self_{i}_in"] = ((NUM_HEADS, 1, MEAN_DECODE_LEN - 1, HEAD_DIM), "float32")
    for i in range(NUM_BLOCKS):
        specs[f"k_cache_cross_{i}"] = ((NUM_HEADS, 1, HEAD_DIM, AUDIO_EMB_LEN), "float32")
        specs[f"v_cache_cross_{i}"] = ((NUM_HEADS, 1, AUDIO_EMB_LEN, HEAD_DIM), "float32")
    specs["position_ids"] = ((1,), "int32")
    return specs


# ---------------------------------------------------------------------------
# qnn-onnx-converter IO dtype 스펙
# ---------------------------------------------------------------------------

def encoder_io_dtypes() -> dict[str, str]:
    # encoder 입력은 w8a16 activation → uint16
    return {"input_features": "uint16"}


def decoder_io_dtypes() -> dict[str, str]:
    """
    qnn-onnx-converter --input_data_type 용 dtype 맵.

    k/v_cache_self_*_in : uint8
        HTP V68 attention 커널이 self-KV cache 입력으로 uint8을 요구.
        AIMET QuantSim 생성 후 compute_encodings() 전에 set_bitwidth(8)을 호출해
        calibration 자체를 8-bit 기준으로 수행하므로 encodings과 dtype이 일관됨.

    attention_mask, k/v_cache_cross_* : uint16 (w8a16 activation)
    input_ids, position_ids : int32 (정수 인덱스, 양자화 없음)
    """
    dtypes: dict[str, str] = {
        "input_ids": "int32",
        "attention_mask": "uint16",
        "position_ids": "int32",
    }
    for i in range(NUM_BLOCKS):
        dtypes[f"k_cache_self_{i}_in"] = "uint8"
        dtypes[f"v_cache_self_{i}_in"] = "uint8"
    for i in range(NUM_BLOCKS):
        dtypes[f"k_cache_cross_{i}"] = "uint16"
        dtypes[f"v_cache_cross_{i}"] = "uint16"
    return dtypes


# ---------------------------------------------------------------------------
# Stage 1a: Encoder — ONNX export + AIMET 양자화
# ---------------------------------------------------------------------------

def export_and_quantize_encoder(
    fp_encoder,
    calib_mels: np.ndarray,
    output_dir: str,
) -> None:
    print("\n========== Encoder: ONNX export + AIMET 양자화 ==========")
    enc_dir = os.path.join(output_dir, "encoder_aimet")
    os.makedirs(enc_dir, exist_ok=True)

    # ONNX export
    spec = encoder_input_spec()
    dummy_input = tuple(make_torch_inputs(spec))
    raw_onnx = os.path.join(enc_dir, "encoder_raw.onnx")
    fp_encoder.eval()
    safe_torch_onnx_export(
        fp_encoder,
        dummy_input,
        raw_onnx,
        export_params=True,
        input_names=list(spec.keys()),
        output_names=HfWhisperEncoder.get_output_names(NUM_BLOCKS),
    )
    print(f"  Raw ONNX → {raw_onnx}")

    # Simplify + CLE
    print("  ONNX 단순화 ...")
    onnx_model, _ = simplify(raw_onnx)
    print("  Cross-layer equalization ...")
    equalize_model(onnx_model)
    os.remove(raw_onnx)

    # QuantSim (w8a16)
    print("  QuantizationSimModel 생성 (w8a16) ...")
    quant_sim = QuantizationSimModel(
        model=onnx_model,
        quant_scheme=QuantScheme.min_max,
        param_type=aimet_onnx.int8,
        activation_type=aimet_onnx.int16,
        config_file=WHISPER_AIMET_CONFIG,
    )

    # Calibration
    input_names = [inp.name for inp in quant_sim.session.get_inputs()]
    calib_data = [{input_names[0]: calib_mels[i]} for i in range(len(calib_mels))]
    print(f"  인코더 캘리브레이션 ({len(calib_data)}개 샘플) ...")
    quant_sim.compute_encodings(calib_data)

    # AIMET 산출물 저장
    quant_sim.export(enc_dir, "model")
    print(f"  AIMET 저장 완료 → {enc_dir}/model.onnx + model.encodings")


# ---------------------------------------------------------------------------
# Stage 1b: Decoder — ONNX export + AIMET 양자화
# ---------------------------------------------------------------------------

def export_and_quantize_decoder(
    fp_encoder,
    fp_decoder,
    calib_mels: np.ndarray,
    output_dir: str,
    num_calib: int = 20,
) -> None:
    print("\n========== Decoder: ONNX export + AIMET 양자화 ==========")
    dec_dir = os.path.join(output_dir, "decoder_aimet")
    os.makedirs(dec_dir, exist_ok=True)

    # ONNX export
    spec = decoder_input_spec()
    dummy_input = tuple(make_torch_inputs(spec))
    raw_onnx = os.path.join(dec_dir, "decoder_raw.onnx")
    fp_decoder.eval()
    safe_torch_onnx_export(
        fp_decoder,
        dummy_input,
        raw_onnx,
        export_params=True,
        input_names=list(spec.keys()),
        output_names=HfWhisperDecoder.get_output_names(NUM_BLOCKS),
    )
    print(f"  Raw ONNX → {raw_onnx}")

    # Simplify + CLE
    print("  ONNX 단순화 ...")
    onnx_model, _ = simplify(raw_onnx)
    print("  Cross-layer equalization ...")
    equalize_model(onnx_model)
    os.remove(raw_onnx)

    # QuantSim (w8a16)
    print("  QuantizationSimModel 생성 (w8a16) ...")
    quant_sim = QuantizationSimModel(
        model=onnx_model,
        quant_scheme=QuantScheme.min_max,
        param_type=aimet_onnx.int8,
        activation_type=aimet_onnx.int16,
        config_file=WHISPER_AIMET_CONFIG,
    )

    # k/v_cache_self_*_in을 8-bit으로 설정 (compute_encodings 전에 호출해야 함)
    #
    # 이유: activation_type=int16이므로 AIMET은 모든 activation (모델 입력 포함)에
    # bw=16을 할당한다. 그러나 QCS6490 HTP V68의 attention 커널은 self-KV cache
    # 입력으로 uint8만 허용한다. compute_encodings() 전에 bitwidth를 8로 변경하면
    # calibration 자체가 8-bit 기준으로 수행되어 scale/offset이 올바르게 계산된다.
    # (JSON 파일을 사후 패치하는 방식은 scale/offset이 16-bit 범위로 계산된 채로
    # bw만 바뀌어 불일치가 생기므로 잘못된 방법이다.)
    changed = 0
    for name, wrapper in quant_sim.qc_quantize_op_dict.items():
        if name.startswith(("k_cache_self_", "v_cache_self_")):
            wrapper.set_bitwidth(8)
            changed += 1
    print(f"  self-KV cache quantizer {changed}개 → bw=8 설정 완료")

    # Decoder calibration data 생성
    # 디코더 입력 중 k/v_cache_self_*_in은 첫 번째 스텝에서 전부 0이므로,
    # 0만으로 calibration하면 AIMET이 scale≈0인 degenerate quantizer를 계산한다.
    # 2-step calibration으로 step 1의 실제 KV 값을 포함시킨다.
    n = min(num_calib, len(calib_mels))
    calib_steps = 2
    print(f"  디코더 캘리브레이션 데이터 생성 ({n}샘플 × {calib_steps}스텝 = {n * calib_steps}개) ...")
    input_names = [inp.name for inp in quant_sim.session.get_inputs()]

    calib_data = []
    fp_encoder.eval()
    fp_decoder.eval()
    with torch.no_grad():
        for i in range(n):
            mel_tensor = torch.from_numpy(calib_mels[i])
            encoder_out = fp_encoder(mel_tensor)
            cross_k = [encoder_out[j][0].cpu() for j in range(NUM_BLOCKS)]
            cross_v = [encoder_out[j][1].cpu() for j in range(NUM_BLOCKS)]
            self_k = [torch.zeros(NUM_HEADS, 1, HEAD_DIM, MEAN_DECODE_LEN - 1) for _ in range(NUM_BLOCKS)]
            self_v = [torch.zeros(NUM_HEADS, 1, MEAN_DECODE_LEN - 1, HEAD_DIM) for _ in range(NUM_BLOCKS)]

            for step in range(calib_steps):
                sample: dict[str, np.ndarray] = {
                    "input_ids": np.array([[SOT_TOKEN_ID]], dtype=np.int32),
                    "position_ids": np.array([step], dtype=np.int32),
                }
                attn_mask = np.full((1, 1, 1, MEAN_DECODE_LEN), MASK_NEG, dtype=np.float32)
                attn_mask[0, 0, 0, MEAN_DECODE_LEN - 1 - step:] = 0.0
                sample["attention_mask"] = attn_mask
                for j in range(NUM_BLOCKS):
                    sample[f"k_cache_self_{j}_in"] = self_k[j].numpy()
                    sample[f"v_cache_self_{j}_in"] = self_v[j].numpy()
                for j in range(NUM_BLOCKS):
                    sample[f"k_cache_cross_{j}"] = cross_k[j].numpy()
                    sample[f"v_cache_cross_{j}"] = cross_v[j].numpy()

                calib_data.append({name: sample[name] for name in input_names})

                # 다음 스텝을 위해 FP 디코더로 실제 KV 값 획득
                dec_args = [torch.tensor(sample["input_ids"]), torch.tensor(sample["attention_mask"])]
                for j in range(NUM_BLOCKS):
                    dec_args.extend([self_k[j], self_v[j]])
                for j in range(NUM_BLOCKS):
                    dec_args.extend([cross_k[j], cross_v[j]])
                dec_args.append(torch.tensor([step], dtype=torch.int32))
                _, kv_new = fp_decoder(*dec_args)
                for j in range(NUM_BLOCKS):
                    self_k[j] = kv_new[j][0].detach()
                    self_v[j] = kv_new[j][1].detach()

    print(f"  디코더 캘리브레이션 ({len(calib_data)}개 샘플) ...")
    quant_sim.compute_encodings(calib_data)

    # AIMET 산출물 저장
    quant_sim.export(dec_dir, "model")
    print(f"  AIMET 저장 완료 → {dec_dir}/model.onnx + model.encodings")


# ---------------------------------------------------------------------------
# Stage 2: QNN SDK 로컬 컴파일
# ---------------------------------------------------------------------------

def _find_tool(sdk_root: str, name: str) -> str:
    """QNN SDK 툴 경로를 반환. SDK 내에 없으면 PATH에서 탐색."""
    path = os.path.join(sdk_root, "bin", "x86_64-linux-clang", name)
    if os.path.isfile(path):
        return path
    fallback = shutil.which(name)
    if fallback:
        return fallback
    raise FileNotFoundError(
        f"QNN 툴 '{name}' 없음.\n"
        f"  예상 경로: {path}\n"
        f"  QNN_SDK_ROOT={sdk_root}"
    )


def _find_lib(sdk_root: str, rel_path: str) -> str:
    path = os.path.join(sdk_root, rel_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"QNN 라이브러리 없음: {path}")
    return path


def _run(cmd: list[str], label: str) -> None:
    print(f"\n  [{label}]")
    print("  $", " ".join(cmd))
    sys.stdout.flush()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  오류: {label} 실패 (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def compile_component(
    sdk_root: str,
    aimet_dir: str,
    out_dir: str,
    label: str,
    input_dims: dict[str, tuple],
    input_dtypes: dict[str, str],
    target: str,
    clean: bool,
) -> None:
    """
    한 컴포넌트(encoder/decoder)의 3-step QNN 컴파일을 수행하고
    최종 산출물(model.onnx + model.bin)을 out_dir에 저장한다.

    Step 1. qnn-onnx-converter
            ONNX + AIMET encodings → model.cpp + model_weights.bin
    Step 2. qnn-model-lib-generator
            model.cpp + model_weights.bin → libmodel.so
    Step 3. qnn-context-binary-generator (x86 오프라인 HTP 컴파일)
            libmodel.so → model.serialized.bin (→ 최종 model.bin)
    """
    onnx_src = os.path.join(aimet_dir, "model.onnx")
    enc_src = os.path.join(aimet_dir, "model.encodings")

    if not os.path.exists(onnx_src):
        print(f"\n  경고: {onnx_src} 없음 – {label} 건너뜀", file=sys.stderr)
        return

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    inter_dir = os.path.join(out_dir, "_intermediate")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    # Step 1: qnn-onnx-converter
    out_cpp = os.path.join(inter_dir, "model.cpp")
    cmd = [
        _find_tool(sdk_root, "qnn-onnx-converter"),
        "--input_network", onnx_src,
        "--output_path", out_cpp,
        "--act_bw", "16",
        "--weight_bw", "8",
    ]
    if os.path.exists(enc_src):
        cmd += ["--quantization_overrides", enc_src]
    else:
        print(f"  경고: encodings 파일 없음 ({enc_src})", file=sys.stderr)
    for name, shape in input_dims.items():
        cmd += ["--input_dim", name, ",".join(str(d) for d in shape)]
    for name, dtype in input_dtypes.items():
        cmd += ["--input_data_type", name, dtype]
    _run(cmd, "qnn-onnx-converter")

    weights_bin = out_cpp.replace(".cpp", ".bin")
    if not os.path.exists(weights_bin):
        raise FileNotFoundError(f"converter 출력 .bin 없음: {weights_bin}")

    # Step 2: qnn-model-lib-generator
    lib_dir = os.path.join(inter_dir, "lib")
    os.makedirs(lib_dir, exist_ok=True)
    _run(
        [_find_tool(sdk_root, "qnn-model-lib-generator"),
         "-c", out_cpp, "-b", weights_bin, "-o", lib_dir, "-t", target],
        "qnn-model-lib-generator",
    )
    so_path = os.path.join(lib_dir, target, "libmodel.so")
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"컴파일된 .so 없음: {so_path}")

    # Step 3: qnn-context-binary-generator (x86 오프라인 HTP 컴파일)
    htp_prepare = _find_lib(sdk_root, "lib/x86_64-linux-clang/libQnnHtpPrepare.so")
    ctx_dir = os.path.join(inter_dir, "ctx")
    os.makedirs(ctx_dir, exist_ok=True)
    _run(
        [_find_tool(sdk_root, "qnn-context-binary-generator"),
         "--backend", htp_prepare,
         "--model", so_path,
         "--output_dir", ctx_dir,
         "--binary_file", "model",
         "--htp_arch_version", str(HTP_ARCH_VERSION),
         "--soc_id", str(SOC_ID_QCS6490)],
        "qnn-context-binary-generator",
    )

    # 컨텍스트 바이너리 탐색 (.bin / .serialized.bin)
    ctx_bin = None
    for candidate in ("model.serialized.bin", "model.bin"):
        path = os.path.join(ctx_dir, candidate)
        if os.path.exists(path):
            ctx_bin = path
            break
    if ctx_bin is None:
        raise FileNotFoundError(
            f"컨텍스트 바이너리 없음: {ctx_dir}/model.serialized.bin 또는 model.bin\n"
            "qnn-context-binary-generator 출력 로그를 확인하세요."
        )

    # model.onnx: qnn-context-binary-generator가 생성한 QNN ONNX 래퍼를 우선 사용.
    # QNN SDK 버전에 따라 생성되지 않을 수 있으므로, 없으면 AIMET ONNX를 복사한다.
    ctx_onnx = os.path.join(ctx_dir, "model.onnx")
    if os.path.exists(ctx_onnx):
        src_onnx = ctx_onnx
        print("  QNN ONNX 래퍼 사용 (qnn-context-binary-generator 생성)")
    else:
        src_onnx = onnx_src
        print("  경고: QNN ONNX 래퍼 없음 → AIMET model.onnx 복사")

    final_onnx = os.path.join(out_dir, "model.onnx")
    final_bin = os.path.join(out_dir, "model.bin")
    shutil.copy2(src_onnx, final_onnx)
    shutil.copy2(ctx_bin, final_bin)

    print(f"\n  [산출물]")
    print(f"    {final_onnx}")
    print(f"    {final_bin}  ({os.path.getsize(final_bin) / 1024 / 1024:.1f} MB)")

    if clean:
        shutil.rmtree(inter_dir, ignore_errors=True)
        print(f"  [--clean] 중간 파일 삭제: {inter_dir}")


def compile_all(
    output_dir: str,
    sdk_root: str,
    target: str,
    encoder_only: bool,
    decoder_only: bool,
    clean: bool,
) -> None:
    print(f"\n========== QNN 로컬 컴파일 (HTP V{HTP_ARCH_VERSION}, QCS6490) ==========")

    enc_dims = {"input_features": (1, 80, MEL_LEN)}

    dec_dims: dict[str, tuple] = {
        "input_ids": (1, 1),
        "attention_mask": (1, 1, 1, MEAN_DECODE_LEN),
        "position_ids": (1,),
    }
    for i in range(NUM_BLOCKS):
        dec_dims[f"k_cache_self_{i}_in"] = (NUM_HEADS, 1, HEAD_DIM, MEAN_DECODE_LEN - 1)
        dec_dims[f"v_cache_self_{i}_in"] = (NUM_HEADS, 1, MEAN_DECODE_LEN - 1, HEAD_DIM)
    for i in range(NUM_BLOCKS):
        dec_dims[f"k_cache_cross_{i}"] = (NUM_HEADS, 1, HEAD_DIM, AUDIO_EMB_LEN)
        dec_dims[f"v_cache_cross_{i}"] = (NUM_HEADS, 1, AUDIO_EMB_LEN, HEAD_DIM)

    if not decoder_only:
        compile_component(
            sdk_root=sdk_root,
            aimet_dir=os.path.join(output_dir, "encoder_aimet"),
            out_dir=os.path.join(output_dir, "encoder_precompiled"),
            label="Encoder → precompiled_qnn_onnx",
            input_dims=enc_dims,
            input_dtypes=encoder_io_dtypes(),
            target=target,
            clean=clean,
        )

    if not encoder_only:
        compile_component(
            sdk_root=sdk_root,
            aimet_dir=os.path.join(output_dir, "decoder_aimet"),
            out_dir=os.path.join(output_dir, "decoder_precompiled"),
            label="Decoder → precompiled_qnn_onnx",
            input_dims=dec_dims,
            input_dtypes=decoder_io_dtypes(),
            target=target,
            clean=clean,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Whisper Small (500-mel/~5s) AIMET w8a16 양자화 + QNN 2.37 로컬 컴파일\n"
            "타겟: QCS6490 (HTP V68)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calib-npz", default=DEFAULT_CALIB_NPZ,
        help="캘리브레이션 NPZ 파일 ('mels' 키, shape (N,1,80,500) float32). "
             f"기본값: {DEFAULT_CALIB_NPZ}",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"출력 디렉토리. 기본값: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--qnn-sdk-root", default=os.environ.get("QNN_SDK_ROOT", ""),
        help="QNN SDK 2.37 루트 경로. 기본값: $QNN_SDK_ROOT",
    )
    parser.add_argument(
        "--target", default="aarch64-android",
        choices=["aarch64-android", "aarch64-oe-linux-gcc11.2", "x86_64-linux-clang"],
        help="qnn-model-lib-generator 컴파일 타겟. 기본값: aarch64-android",
    )
    parser.add_argument(
        "--decoder-calib-samples", type=int, default=20,
        help="디코더 캘리브레이션에 사용할 mel 샘플 수. 기본값: 20",
    )
    parser.add_argument(
        "--skip-compile", action="store_true",
        help="양자화만 수행하고 QNN 컴파일은 생략",
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="기존 AIMET 산출물로 QNN 컴파일만 수행",
    )
    parser.add_argument(
        "--encoder-only", action="store_true",
        help="인코더만 처리",
    )
    parser.add_argument(
        "--decoder-only", action="store_true",
        help="디코더만 처리",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="컴파일 완료 후 _intermediate/ 중간 파일 삭제",
    )
    args = parser.parse_args()

    if args.skip_compile and args.compile_only:
        parser.error("--skip-compile과 --compile-only는 함께 사용 불가")
    if args.encoder_only and args.decoder_only:
        parser.error("--encoder-only와 --decoder-only는 함께 사용 불가")

    if not args.skip_compile:
        if not args.qnn_sdk_root:
            parser.error(
                "QNN SDK 경로가 필요합니다.\n"
                "  export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.37.0.240927\n"
                "  또는 --qnn-sdk-root 인수 사용"
            )
        if not os.path.isdir(args.qnn_sdk_root):
            parser.error(f"QNN SDK 디렉토리 없음: {args.qnn_sdk_root}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: AIMET 양자화
    if not args.compile_only:
        print(f"캘리브레이션 데이터 로드: {args.calib_npz}")
        calib_mels = np.load(args.calib_npz)["mels"]  # (N, 1, 80, 500) float32
        print(f"  shape={calib_mels.shape}  dtype={calib_mels.dtype}")

        print("\nWhisperSmall FP 모델 로드 ...")
        whisper = WhisperSmall.from_pretrained()
        fp_encoder = whisper.encoder
        fp_decoder = whisper.decoder

        # Positional embedding trim: (1500, 768) → (250, 768)
        orig_shape = fp_encoder.encoder.embed_positions.shape
        fp_encoder.encoder.embed_positions = nn.Parameter(
            fp_encoder.encoder.embed_positions[:AUDIO_EMB_LEN, :]
        )
        print(f"  embed_positions: {orig_shape} → {fp_encoder.encoder.embed_positions.shape}")

        if not args.decoder_only:
            export_and_quantize_encoder(fp_encoder, calib_mels, args.output_dir)
        if not args.encoder_only:
            export_and_quantize_decoder(
                fp_encoder, fp_decoder, calib_mels, args.output_dir,
                num_calib=args.decoder_calib_samples,
            )

    # Stage 2: QNN 로컬 컴파일
    if not args.skip_compile:
        compile_all(
            output_dir=args.output_dir,
            sdk_root=args.qnn_sdk_root,
            target=args.target,
            encoder_only=args.encoder_only,
            decoder_only=args.decoder_only,
            clean=args.clean,
        )
    else:
        print("\n  --skip-compile: QNN 컴파일 생략. AIMET 산출물만 생성됨.")

    print(f"\n완료. 출력 디렉토리: {args.output_dir}")
    if not args.skip_compile:
        if not args.decoder_only:
            print(f"  encoder_precompiled/model.onnx + model.bin")
        if not args.encoder_only:
            print(f"  decoder_precompiled/model.onnx + model.bin")


if __name__ == "__main__":
    main()
