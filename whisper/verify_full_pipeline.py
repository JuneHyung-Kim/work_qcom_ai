import whisper
import torch
import onnxruntime as ort
import numpy as np
import time
import os

class OnnxEncoderWrapper(torch.nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # Keep track of the original device/dtype from a dummy parameter if needed, 
        # but here we just need to return a compatible tensor.

    def forward(self, x):
        # x shape: [batch, 80, 500] (or similar)
        # Convert torch tensor to numpy
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        # Ensure input is 500 frames (pad or truncate if necessary, though logic should happen before)
        # For this specific wrapper, we assume the input is correctly shaped to roughly what the ONNX expects 
        # OR we just pass it blindly if the caller ensures shape.
        
        numpy_input = x.cpu().numpy().astype(np.float32)
        
        start = time.time()
        result_np = self.session.run([self.output_name], {self.input_name: numpy_input})[0]
        # result_np shape: [batch, n_frames, n_model] -> likely [1, 250, 384]
        
        # Convert back to torch tensor
        # The decoder expects the encoder output to be on the same device as the decoder.
        # We need to check where the original model is, but for 'cpu' it's fine.
        result_tensor = torch.from_numpy(result_np)
        
        return result_tensor

def verify_full_pipeline():
    model_name = "small"
    onnx_model_path = "whisper_encoder_5s_fp32.onnx"
    samples = ["mel_hello_english.npy", "mel_hello_korean.npy"]

    print(f"Loading Original Whisper Model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Loading 5s ONNX Encoder: {onnx_model_path}")
    onnx_encoder = OnnxEncoderWrapper(onnx_model_path)

    # Store original encoder to swap back and forth or just use a separate instance if memory allows. 
    # For "small", one instance is fine, we can hot-swap the encoder.
    original_encoder = model.encoder

    for sample_path in samples:
        if not os.path.exists(sample_path):
            print(f"Sample not found: {sample_path}")
            continue

        print(f"\n--- Processing {sample_path} ---")
        mel = np.load(sample_path)
        # Mel shape verification
        # Whisper load_audio -> log_mel_spectrogram usually gives [80, T]
        # Our saved mels might be [80, 3000] ready for the original or raw [80, Actual_T]
        
        print(f"Loaded Mel Shape: {mel.shape}")
        
        # 1. Original Pipeline (30s)
        print("Running Original Pipeline (padded to 30s)...")
        
        # Prepare 30s input
        # Standard whisper.pad_or_trim defaults to N_SAMPLES (480000) for audio,
        # but we need N_FRAMES (3000) for Mel spectrograms.
        mel_30s = whisper.pad_or_trim(mel, length=3000)
        mel_30s_tensor = torch.from_numpy(mel_30s).type(torch.float32).to(model.device).unsqueeze(0)
        
        print(f"Debug: mel_30s_tensor shape: {mel_30s_tensor.shape}")
        
        # Ensure we are using original encoder
        model.encoder = original_encoder
        
        start_orig = time.time()
        # Decode
        # We use a decode option for reproducible results (greedy)
        options = whisper.DecodingOptions(fp16=False)
        try:
            result_orig = whisper.decode(model, mel_30s_tensor, options)
            if isinstance(result_orig, list):
                result_orig = result_orig[0]
            end_orig = time.time()
            
            print(f"Original Text: {result_orig.text}")
            print(f"Original Time: {end_orig - start_orig:.4f}s")
        except Exception as e:
            print(f"Original Pipeline Failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue to test Hybrid anyway if possible, or just skip
            continue
        
        # 2. Hybrid Pipeline (5s)
        print("Running Hybrid Pipeline (5s ONNX Encoder)...")
        
        # Prepare 5s input
        target_len = 500
        current_len = mel.shape[1]
        
        if current_len < target_len:
            pad_amount = target_len - current_len
            mel_5s = np.pad(mel, ((0,0), (0, pad_amount)), mode='constant')
        else:
            mel_5s = mel[:, :target_len]
            
        mel_5s_tensor = torch.from_numpy(mel_5s).to(model.device).unsqueeze(0)
        
        # Swap encoder
        model.encoder = onnx_encoder
        
        # Important: The decoder handles variable length context in cross-attention?
        # Yes, standard transformer decoder architecture attends to whatever Key/Value length comes from encoder.
        
        start_5s = time.time()
        try:
            result_5s = whisper.decode(model, mel_5s_tensor, options)
            if isinstance(result_5s, list):
                result_5s = result_5s[0]
            end_5s = time.time()
            
            print(f"Hybrid Text: {result_5s.text}")
            print(f"Hybrid Time: {end_5s - start_5s:.4f}s")
            
            if result_orig.text.strip() == result_5s.text.strip():
                 print("SUCCESS: Outputs match exactly.")
            else:
                 print("WARNING: Outputs differ.")
        except Exception as e:
            print(f"Hybrid Pipeline Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_full_pipeline()
