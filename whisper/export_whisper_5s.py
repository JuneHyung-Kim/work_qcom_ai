import torch
import whisper
import warnings

# Suppress minor warnings
warnings.filterwarnings("ignore")

def export_whisper_encoder_5s(output_path="whisper_encoder_5s_fp32.onnx"):
    print("Loading Whisper Small model...")
    # Load the PyTorch model
    model = whisper.load_model("small")
    model.encoder.eval()

    print("Creating dummy input [1, 80, 500]...")
    # 5 seconds of audio corresponds to 500 frames (10ms per frame)
    # Whisper expects Mel spectrogram input: [Batch, n_mels, n_frames]
    dummy_input = torch.randn(1, 80, 500)

    # Truncate positional embedding to match the 5s input (500 frames)
    # Whisper encoder has a stride of 2, so 500 frames -> 250 embeddings
    # Original shape: [1500, 384] -> New shape: [250, 384]
    new_pos_embed = model.encoder.positional_embedding[:250, :]
    model.encoder.positional_embedding = torch.nn.Parameter(new_pos_embed)

    print(f"Exporting to {output_path}...")
    try:
        torch.onnx.export(
            model.encoder,
            dummy_input,
            output_path,
            input_names=["mels"],
            output_names=["output_features"],
            opset_version=14,  # Recommended opset for modern transformers
            dynamic_axes=None  # Fixed input size
        )
        print(f"Successfully exported to {output_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False

if __name__ == "__main__":
    export_whisper_encoder_5s()
