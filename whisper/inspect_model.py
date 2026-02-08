import onnxruntime as ort
import sys

def inspect_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        print(f"Model: {model_path}")
        print("Inputs:")
        for input_meta in session.get_inputs():
            print(f"  Name: {input_meta.name}")
            print(f"  Shape: {input_meta.shape}")
            print(f"  Type: {input_meta.type}")
        
        print("Outputs:")
        for output_meta in session.get_outputs():
            print(f"  Name: {output_meta.name}")
            print(f"  Shape: {output_meta.shape}")
            print(f"  Type: {output_meta.type}")
            
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <model_path>")
    else:
        inspect_model(sys.argv[1])
