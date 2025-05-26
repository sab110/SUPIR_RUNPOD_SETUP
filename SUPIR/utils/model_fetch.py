from huggingface_hub import snapshot_download
import os

# Use the workspace models directory for RunPod
models_folder = os.getenv('MODELS_DIR', '/workspace/models')
if not os.path.exists(models_folder):
    models_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", 'models'))


def get_model(model_repo: str):
    # Handle local paths (e.g., 'models/clip-vit-large-patch14')
    if model_repo.startswith('models/'):
        # Remove 'models/' prefix and use the workspace models directory
        model_name = model_repo.replace('models/', '')
        model_path = os.path.join(models_folder, model_name)
        
        if os.path.exists(model_path):
            print(f"Found local model: {model_path}")
            return model_path
        else:
            print(f"Local model not found: {model_path}")
            # Try to download from HuggingFace as fallback
            try:
                # Convert to proper HuggingFace repo ID
                if model_name == 'clip-vit-large-patch14':
                    hf_repo = 'openai/clip-vit-large-patch14'
                elif model_name == 'clip-vit-large-patch14-336':
                    hf_repo = 'openai/clip-vit-large-patch14-336'
                else:
                    hf_repo = model_repo
                
                print(f"Attempting to download from HuggingFace: {hf_repo}")
                snapshot_download(hf_repo, local_dir=model_path, local_dir_use_symlinks=False)
                return model_path
            except Exception as e:
                print(f"Failed to download model {hf_repo}: {e}")
                # Return the path anyway, maybe the model will be found later
                return model_path
    else:
        # Handle HuggingFace repo IDs
        model_name = model_repo.split('/')[-1]
        model_path = os.path.join(models_folder, model_name)
        if not os.path.exists(model_path):
            model_folder = model_repo.split('/')[1]
            snapshot_download(model_repo, local_dir=os.path.join(models_folder, model_folder), local_dir_use_symlinks=False)
        return model_path
