import os
import sys
import torch
import numpy as np
from pathlib import Path
import qai_hub as hub
from qai_hub_models.utils.base_model import Precision, TargetRuntime

# --- 1. Patch Constants ---
# Import the module where constants are defined
import qai_hub_models.models._shared.hf_whisper.model as hf_whisper_params

print(f"Old CHUNK_LENGTH: {hf_whisper_params.CHUNK_LENGTH}")
print("Patching Whisper constants for 5s input...")

# 5 seconds
hf_whisper_params.CHUNK_LENGTH = 5
hf_whisper_params.AUDIO_EMB_LEN = 250 
hf_whisper_params.MELS_AUDIO_LEN = 500
hf_whisper_params.N_SAMPLES = 80000

print(f"New CHUNK_LENGTH: {hf_whisper_params.CHUNK_LENGTH}")

# --- Imports after patching ---
# Use FLOAT models to avoid AIMET dependency on Windows
from qai_hub_models.models._shared.hf_whisper.model import (
    HfWhisperEncoder,
    HfWhisperDecoder,
    HfWhisper,
)
from transformers import WhisperConfig, AutoConfig

WHISPER_VERSION = "openai/whisper-small"

def export_5s_model():
    print("Initializing Float Components (aiming for QNN Quantization)...")
    
    # 1. Encoder
    print("Loading Encoder...")
    # Instantiate Base/Float Encoder
    encoder = HfWhisperEncoder.from_pretrained(WHISPER_VERSION)
    
    # HACK: Slice positional embeddings to match the new 5s input length (250)
    # The pre-trained model has 1500 positions.
    print(f"Slicing encoder positional embeddings to {hf_whisper_params.AUDIO_EMB_LEN}...")
    new_pos_len = hf_whisper_params.AUDIO_EMB_LEN
    # encoder.encoder is the QcWhisperEncoder
    # embed_positions is a Parameter
    current_pos_weights = encoder.encoder.embed_positions.data
    encoder.encoder.embed_positions = torch.nn.Parameter(
        current_pos_weights[:new_pos_len, :]
    )
    
    # Verify shape
    try:
        spec = encoder.get_input_spec()
        print(f"Encoder Input Spec: {spec}")
        if spec['input_features'][0] != (1, 80, 500):
             print(f"WARNING: Encoder input shape mismatch! Expected (1, 80, 500), got {spec['input_features'][0]}")
    except Exception as e:
        print(f"Could not verify encoder spec: {e}")

    # 2. Decoder
    print("Loading Decoder...")
    decoder = HfWhisperDecoder.from_pretrained(WHISPER_VERSION)
    
    # 4. Compile / Export
    device_name = "QCS6490 (Proxy)"
    device = None
    try:
        if hub.get_devices():
            device = hub.Device(device_name)
    except Exception as e:
        print(f"Warning: Could not connect to AI Hub or find device: {e}")
        print("Will proceed with local ONNX export only.")

    output_base_path = Path("export_whisper_5s_output")
    output_base_path.mkdir(exist_ok=True)
    
    # --- Encoder ---
    print("\n--- Processing Encoder ---")
    
    # Export ONNX locally
    enc_input_spec = encoder.get_input_spec()
    print("Exporting Encoder to ONNX...")
    # convert_to_hub_source_model exports to ONNX (or torchscript) at output_path/model_name/
    enc_source_model = encoder.convert_to_hub_source_model(
        target_runtime=TargetRuntime.QNN_CONTEXT_BINARY,
        output_path=output_base_path,
        input_spec=enc_input_spec,
        model_name="whisper_small_5s_encoder"
    )
    print(f"Encoder source model saved to: {output_base_path}/whisper_small_5s_encoder")

    if device:
        print("Submitting Compile Job for Encoder...")
        model_compile_options_enc = encoder.get_hub_compile_options(
            target_runtime=TargetRuntime.QNN_CONTEXT_BINARY,
            precision=Precision.w8a16, 
            device=device
        )
        print(f"Encoder Compile Options: {model_compile_options_enc}")
        
        try:
            enc_job = hub.submit_compile_job(
                model=enc_source_model,
                input_specs=enc_input_spec,
                device=device,
                name="whisper_small_5s_encoder",
                options=model_compile_options_enc
            )
            print(f"Encoder Job Submitted: {enc_job.url}")
        except Exception as e:
             print(f"Failed to submit encoder job: {e}")

    # --- Decoder ---
    print("\n--- Processing Decoder ---")
    
    dec_input_spec = decoder.get_input_spec()
    print("Exporting Decoder to ONNX...")
    dec_source_model = decoder.convert_to_hub_source_model(
        target_runtime=TargetRuntime.QNN_CONTEXT_BINARY,
        output_path=output_base_path,
        input_spec=dec_input_spec,
        model_name="whisper_small_5s_decoder"
    )
    print(f"Decoder source model saved to: {output_base_path}/whisper_small_5s_decoder")

    if device:
        print("Submitting Compile Job for Decoder...")
        model_compile_options_dec = decoder.get_hub_compile_options(
            target_runtime=TargetRuntime.QNN_CONTEXT_BINARY,
            precision=Precision.w8a16,
            device=device
        )
        try:
            dec_job = hub.submit_compile_job(
                model=dec_source_model,
                input_specs=dec_input_spec,
                device=device,
                name="whisper_small_5s_decoder",
                options=model_compile_options_dec
            )
            print(f"Decoder Job Submitted: {dec_job.url}")
        except Exception as e:
            print(f"Failed to submit decoder job: {e}")
            
    if not device:
        print("\n[IMPORTANT] AI Hub not configured. ONNX files were generated locally.")
        print("To run QNN compilation, please configure 'qai-hub configure' or provide API token.")

if __name__ == "__main__":
    export_5s_model()
