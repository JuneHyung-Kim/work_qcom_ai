#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pipeline:
  1) Load FP32 ONNX (Whisper encoder 5s: input [1,80,500])
  2) Static PTQ to QDQ ONNX for QNN EP (onnxruntime.quantization.execution_providers.qnn)
  3) Create ORT session with QNN EP + context dump enabled
     -> generates *_ctx.onnx + QNN context .bin (or embedded if requested)

Usage examples:
  # (A) x86_64: generate QDQ + dump ctx (if QNN EP backend libs are available on this host)
  python build_whisper5s_qnn_ctx.py \
    --fp32_onnx whisper_encoder_5s_fp32.onnx \
    --calib_npz calib_out/calib_whisper_5s_enko.npz \
    --out_dir out_qnn \
    --backend_path /path/to/libQnnHtp.so

  # (B) two-stage practical workflow:
  #   1) x86_64: QDQ only
  python build_whisper5s_qnn_ctx.py ... --skip_ctx_dump
  #   2) target device: ctx dump only (copy qdq.onnx + script + backend libs)
  python build_whisper5s_qnn_ctx.py ... --skip_qdq --qdq_onnx out_qnn/whisper_encoder_5s_fp32.qdq.onnx --backend_path ...

Notes:
- QDQ generation typically needs x86_64. (ORT QNN EP quantization doc)
- Context dump generates:
    <ctx_file_path> (default: <model>_ctx.onnx)
    <model>QNN<hash>.bin   (filename is auto-generated; don't rename/move casually)
"""

import argparse
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import onnxruntime as ort

# ORT quantization imports
from onnxruntime.quantization import CalibrationDataReader, quantize

# QNN EP specific helpers (ORT provides these under quantization.execution_providers.qnn)
try:
    from onnxruntime.quantization.execution_providers.qnn import (
        get_qnn_qdq_config,
        qnn_preprocess_model,
    )
except Exception as e:
    get_qnn_qdq_config = None
    qnn_preprocess_model = None


# -----------------------------
# Calibration DataReader (NPZ: mels shape (N,1,80,500))
# -----------------------------
class NpzMelCalibDataReader(CalibrationDataReader):
    def __init__(self, npz_path: str, input_name: str, expected_shape=(1, 80, 500), dtype=np.float32):
        self.npz_path = npz_path
        self.input_name = input_name
        self.expected_shape = tuple(expected_shape)
        self.dtype = dtype

        data = np.load(npz_path)
        if "mels" not in data:
            raise ValueError(f"NPZ must contain key 'mels'. keys={list(data.keys())}")
        mels = data["mels"]

        if mels.dtype != np.float32:
            mels = mels.astype(np.float32)

        # Expected: (N,1,80,500)
        if mels.ndim != 4:
            raise ValueError(f"mels must be rank-4 (N,1,80,500). got shape={mels.shape}")
        if tuple(mels.shape[1:]) != self.expected_shape:
            raise ValueError(f"mels per-sample shape must be {self.expected_shape}. got {mels.shape[1:]}")

        # store
        self.mels = mels
        self._i = 0

    def get_next(self):
        if self._i >= self.mels.shape[0]:
            return None
        sample = self.mels[self._i]  # (1,80,500)
        self._i += 1
        return {self.input_name: sample.astype(self.dtype, copy=False)}

    def rewind(self):
        self._i = 0


# -----------------------------
# Helpers
# -----------------------------
def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_exists(p: Path, what: str):
    if not p.exists():
        die(f"Missing {what}: {p.resolve()}")


def infer_single_input_name(onnx_path: str) -> str:
    # CPU EP is enough to inspect IO metadata
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    if len(ins) != 1:
        names = [i.name for i in ins]
        die(f"Expected exactly 1 input. got {len(ins)} inputs: {names}")
    return ins[0].name


def qdq_quantize_for_qnn(fp32_onnx: str, calib_npz: str, qdq_out: str, input_name: str):
    if get_qnn_qdq_config is None or qnn_preprocess_model is None:
        die(
            "Failed to import QNN quantization helpers:\n"
            "  from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model\n"
            "Please install an onnxruntime build that includes these modules (often onnxruntime / onnxruntime-gpu, version-dependent)."
        )

    print(f"[QDQ] preprocess model for QNN: {fp32_onnx}")
    preprocessed = qnn_preprocess_model(fp32_onnx)

    print(f"[QDQ] load calib npz: {calib_npz}")
    dr = NpzMelCalibDataReader(
        npz_path=calib_npz,
        input_name=input_name,
        expected_shape=(1, 80, 500),
        dtype=np.float32,
    )

    print("[QDQ] build QNN QDQ config")
    qnn_config = get_qnn_qdq_config(
        preprocessed,
        dr,
        # 기본값으로 시작하는 것을 권장합니다.
        # 필요 시 여기서 op_types_to_quantize / per_channel 등 커스터마이즈 가능
    )

    print(f"[QDQ] quantize -> {qdq_out}")
    Path(qdq_out).parent.mkdir(parents=True, exist_ok=True)
    quantize(preprocessed, qdq_out, qnn_config)
    print("[QDQ] done")


def dump_qnn_ep_context(qdq_onnx: str, ctx_onnx_out: str, backend_path: str, embed: bool, disable_cpu_fallback: bool):
    so = ort.SessionOptions()

    # EP Context dump options (SessionOptions config entries)
    # - ep.context_enable: 1 => dump
    # - ep.context_file_path: output path for *_ctx.onnx
    # - ep.context_embed_mode: 1 => embed bin into onnx
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", str(ctx_onnx_out))
    if embed:
        so.add_session_config_entry("ep.context_embed_mode", "1")

    # (권장) 미지원 op가 CPU로 fallback 되는 것을 막아, "완전 오프로딩" 상태를 확인
    if disable_cpu_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    providers = ["QNNExecutionProvider"]
    provider_options = [{"backend_path": backend_path}]

    print("[CTX] create ORT session with QNN EP (context dump enabled)")
    print(f"[CTX] model : {qdq_onnx}")
    print(f"[CTX] ctx   : {ctx_onnx_out}")
    print(f"[CTX] backend_path: {backend_path}")
    print(f"[CTX] embed_bin_in_onnx: {embed}")
    print(f"[CTX] disable_cpu_ep_fallback: {disable_cpu_fallback}")

    # NOTE: context dump happens during session creation; no inference run required.
    sess = ort.InferenceSession(qdq_onnx, sess_options=so, providers=providers, provider_options=provider_options)
    _ = sess.get_inputs()  # touch metadata

    print("[CTX] session created. context dump should be generated if QNN EP succeeded.")
    print("[CTX] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_onnx", type=str, default="whisper_encoder_5s_fp32.onnx")
    ap.add_argument("--calib_npz", type=str, default="calib_out/calib_whisper_5s_enko.npz")
    ap.add_argument("--out_dir", type=str, default="out_qnn")

    ap.add_argument("--input_name", type=str, default="", help="ONNX input name (default: auto-detect)")
    ap.add_argument("--backend_path", type=str, default="", help="Path to libQnnHtp.so (or appropriate backend lib)")

    ap.add_argument("--qdq_onnx", type=str, default="", help="If provided with --skip_qdq, use this qdq model")
    ap.add_argument("--ctx_onnx", type=str, default="", help="Output ctx onnx path (default: <out_dir>/<qdq_basename>_ctx.onnx)")

    ap.add_argument("--embed", action="store_true", help="Embed QNN context binary into ctx onnx (ep.context_embed_mode=1)")
    ap.add_argument("--disable_cpu_fallback", action="store_true", help="Disable CPU EP fallback to enforce full QNN partitioning")

    ap.add_argument("--skip_qdq", action="store_true", help="Skip QDQ generation step")
    ap.add_argument("--skip_ctx_dump", action="store_true", help="Skip ctx dump step")

    args = ap.parse_args()

    fp32_onnx = Path(args.fp32_onnx)
    calib_npz = Path(args.calib_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_qdq:
        ensure_exists(fp32_onnx, "fp32_onnx")
        ensure_exists(calib_npz, "calib_npz")

    # input name
    if args.input_name:
        input_name = args.input_name
    else:
        if args.skip_qdq:
            # If skipping QDQ, try to infer from qdq model
            if not args.qdq_onnx:
                die("--skip_qdq requires --qdq_onnx (or provide --input_name explicitly and fp32_onnx to infer)")
            input_name = infer_single_input_name(args.qdq_onnx)
        else:
            input_name = infer_single_input_name(str(fp32_onnx))
    print(f"[INFO] input_name: {input_name}")

    # QDQ output path
    if args.qdq_onnx:
        qdq_onnx = Path(args.qdq_onnx)
    else:
        qdq_onnx = out_dir / (fp32_onnx.stem + ".qdq.onnx")

    # 1) QDQ generation
    if not args.skip_qdq:
        print(f"[STEP] QDQ generation start (ORT={ort.__version__})")
        qdq_quantize_for_qnn(
            fp32_onnx=str(fp32_onnx),
            calib_npz=str(calib_npz),
            qdq_out=str(qdq_onnx),
            input_name=input_name,
        )
        ensure_exists(qdq_onnx, "qdq_onnx_out")

    # 2) Context dump
    if not args.skip_ctx_dump:
        ensure_exists(qdq_onnx, "qdq_onnx for ctx dump")

        if not args.backend_path:
            die("--backend_path is required for ctx dump (e.g., path to libQnnHtp.so)")
        backend_path = args.backend_path

        if args.ctx_onnx:
            ctx_onnx = Path(args.ctx_onnx)
        else:
            ctx_onnx = out_dir / (qdq_onnx.stem + "_ctx.onnx")

        ctx_onnx.parent.mkdir(parents=True, exist_ok=True)

        print("[STEP] QNN EP context dump start")
        dump_qnn_ep_context(
            qdq_onnx=str(qdq_onnx),
            ctx_onnx_out=str(ctx_onnx),
            backend_path=backend_path,
            embed=args.embed,
            disable_cpu_fallback=args.disable_cpu_fallback,
        )

        # Post-check: confirm ctx onnx exists
        if not ctx_onnx.exists():
            die(f"Context ONNX was not created: {ctx_onnx.resolve()}\n"
                f"Possible causes: QNN EP not properly loaded, backend_path invalid, partitioning failure with disable_cpu_fallback, etc.")

        print(f"[OK] ctx onnx: {ctx_onnx.resolve()}")
        print("[NOTE] QNN context .bin is auto-generated next to ctx onnx or in working directory depending on ORT build/options.")
        print("       Do not rename/move the generated *.bin casually; the ctx onnx references it (unless embed mode was used).")

    print("[DONE]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
    except Exception as e:
        print("[FATAL] Unhandled exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
