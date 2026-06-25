import torch
import sys

def run_parameter_audit(model_path, max_allowed=50000):
    try:
        # 1. Safely reconstruct the saved model object back into memory
        # weights_only=False allows loading full custom model architectures
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()

        # 2. Extract and sum the exact element counts of all parameters requiring gradients
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n==========================================")
        print(f"📋 AUDIT REPORT FOR: {model_path}")
        print(f"🔢 Total Trainable Parameters: {total_params:,}")
        print("==========================================")

        if total_params > max_allowed:
            print(f"❌ DISQUALIFIED! Model exceeds the budget limit by {total_params - max_allowed:,} parameters.")
            return False

        print(f"✅ PASSED! Model is highly optimized and within the {max_allowed:,} efficiency budget.")
        return True

    except Exception as e:
        print(f"⚠️ Error reading model file: {e}")
        print("Make sure the student submitted a valid saved PyTorch model file.")
        return False

if __name__ == "__main__":
    # Fallback helper if you just double-click the file
    file_to_check = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    run_parameter_audit(file_to_check)