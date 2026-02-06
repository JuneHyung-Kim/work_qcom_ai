import numpy as np
import onnxruntime as ort
import os

def verify_sample(model_path, sample_path):
    print(f"--- Verifying {os.path.basename(sample_path)} ---")
    
    # Load Mel Spectrogram
    mel = np.load(sample_path)
    print(f"Original Mel shape: {mel.shape}")
    
    # Expected shape: [1, 80, 500]
    # Whisper mels are usually [80, T] or [1, 80, T]
    
    if len(mel.shape) == 2:
        mel = mel[np.newaxis, :, :] # Add batch dim -> [1, 80, T]
    
    if mel.shape[1] != 80:
        print(f"Error: Expected 80 mel channels, got {mel.shape[1]}")
        return

    current_width = mel.shape[2]
    target_width = 500
    
    # Pad to 500
    if current_width < target_width:
        padding = target_width - current_width
        mel = np.pad(mel, ((0, 0), (0, 0), (0, padding)), mode='constant')
        print(f"Padded to: {mel.shape}")
    # Truncate to 500
    elif current_width > target_width:
        mel = mel[:, :, :target_width]
        print(f"Truncated to: {mel.shape}")
    
    # Run Inference
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        result = session.run([output_name], {input_name: mel.astype(np.float32)})
        print("Inference Successful!")
        print(f"Output shape: {result[0].shape}")
        
    except Exception as e:
        print(f"Inference Failed: {e}")

if __name__ == "__main__":
    model_path = "whisper_encoder_5s_fp32.onnx"
    samples = ["mel_hello_english.npy", "mel_hello_korean.npy"]
    
    for sample in samples:
        if os.path.exists(sample):
            verify_sample(model_path, sample)
        else:
            print(f"Sample file not found: {sample}")
