import asyncio
import os
import sys
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import traceback
from pathlib import Path

# Add current directory to Python path for SUPIR modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
from PIL import Image
import numpy as np

# Import SUPIR modules
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, HWC3, upscale_image
from SUPIR.utils.status_container import StatusContainer, MediaData
from llava.llava_agent import LLavaAgent
from ui_helpers import printt

# Environment configuration
LOADING_MODE = os.getenv("LOADING_MODE", "fp16")  # Force fp16 for consistency
TILED_VAE = os.getenv("TILED_VAE", "True").lower() == "true"
AUTO_MOVE_CPU = os.getenv("AUTO_MOVE_CPU", "True").lower() == "true"

# API Configuration
API_TOKEN = os.getenv("API_TOKEN", "your-secret-token-here")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
MODELS_DIR = os.getenv("MODELS_DIR", "/workspace/models")
BATCH_INPUT_DIR = os.path.join(WORKSPACE_DIR, "adjusted")
BATCH_OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "adjustedupscaled")

# Ensure directories exist
os.makedirs(BATCH_INPUT_DIR, exist_ok=True)
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="SUPIR API", version="1.0.0", description="REST API for SUPIR image upscaling")
security = HTTPBearer()

# Job status enum
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Request/Response models
class JobSettings(BaseModel):
    upscale_size: int = Field(default=2, ge=1, le=8, description="Upscale factor")
    apply_llava: bool = Field(default=True, description="Apply LLaVa caption generation")
    apply_supir: bool = Field(default=True, description="Apply SUPIR upscaling")
    prompt_style: str = Field(default="Photorealistic", description="Prompt style")
    model: str = Field(default="v0F.ckpt", description="Model checkpoint")
    checkpoint_type: str = Field(default="Standard SDXL", description="Checkpoint type")
    prompt: str = Field(default="", description="Custom prompt")
    save_captions: bool = Field(default=True, description="Save generated captions")
    text_guidance_scale: int = Field(default=1024, ge=1, le=2048, description="Text guidance scale")
    background_restoration: bool = Field(default=True, description="Enable background restoration")
    face_restoration: bool = Field(default=True, description="Enable face restoration")
    edm_steps: int = Field(default=50, ge=1, le=200, description="EDM sampling steps")
    s_stage1: float = Field(default=1.0, ge=0.0, le=2.0, description="Stage 1 strength")
    s_stage2: float = Field(default=1.0, ge=0.0, le=2.0, description="Stage 2 strength")
    s_cfg: float = Field(default=7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    sampler: str = Field(default="DPMPP2M", description="Sampler type")
    s_churn: float = Field(default=0, ge=0.0, le=1.0, description="Churn parameter")
    s_noise: float = Field(default=1.003, ge=0.0, le=2.0, description="Noise parameter")
    color_fix_type: str = Field(default="Wavelet", description="Color fix type")
    linear_cfg: bool = Field(default=False, description="Use linear CFG")
    linear_s_stage2: bool = Field(default=False, description="Use linear stage 2")

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    message: str = ""

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""
    error: Optional[str] = None

# Global variables for model management
supir_model = None
llava_agent = None
jobs: Dict[str, Dict[str, Any]] = {}

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Model loading functions
def load_supir_model(model_name: str, checkpoint_type: str):
    global supir_model
    
    try:
        printt(f"Loading SUPIR model: {model_name}")
        
        # Determine config path based on checkpoint type
        if "SDXL" in checkpoint_type:
            config_path = "options/SUPIR_v0.yaml"
        else:
            config_path = "options/SUPIR_v0.yaml"  # Default fallback
        
        # Determine weight dtype based on LOADING_MODE
        weight_dtype = LOADING_MODE.lower()
        if weight_dtype not in ['fp16', 'fp32', 'bf16']:
            weight_dtype = 'fp16'
        
        # Force fp16 to avoid mixed precision issues
        if weight_dtype == 'bf16':
            weight_dtype = 'fp16'
        
        # Model checkpoint path
        ckpt_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        printt(f"CUDA available: {torch.cuda.is_available()}")
        printt(f"Creating SUPIR model with config: {config_path}, weight_dtype: {weight_dtype}, device: {device}, ckpt: {ckpt_path}")
        
        # Map sampler names to full module paths
        sampler_mapping = {
            "DPMPP2M": "sgm.modules.diffusionmodules.sampling.RestoreDPMPP2MSampler",
            "DDIM": "sgm.modules.diffusionmodules.sampling.DDIMSampler",
            "DPM": "sgm.modules.diffusionmodules.sampling.DPMSampler"
        }
        
        # Get the full sampler path, default to the original config if not found
        sampler_target = sampler_mapping.get("DPMPP2M", "sgm.modules.diffusionmodules.sampling.RestoreDPMPP2MSampler")
        
        supir_model = create_SUPIR_model(
            config_path=config_path,
            weight_dtype='fp16',  # Force fp16 for consistency
            device=device,
            ckpt=ckpt_path,
            sampler=sampler_target
        )
        
        # Check model dtypes and device placement
        printt(f"Model ae_dtype: {supir_model.ae_dtype}")
        printt(f"Model diffusion_dtype: {supir_model.model.dtype}")
        
        # Ensure all model components are on the same device
        printt("Ensuring all model components are on CUDA...")
        supir_model.to(device)
        if hasattr(supir_model, 'first_stage_model'):
            supir_model.first_stage_model.to(device)
            if hasattr(supir_model.first_stage_model, 'quant_conv'):
                supir_model.first_stage_model.quant_conv.to(device)
                printt("Moved quant_conv to CUDA")
        printt("All model components moved to CUDA")
        
        printt(f"SUPIR model created successfully")
        
        # Initialize tiled VAE if enabled
        if TILED_VAE:
            printt("Tiled VAE is enabled but temporarily disabled for debugging")
            # supir_model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
        else:
            printt("Tiled VAE is disabled")
        
        # Move to device using the model's custom method
        supir_model.move_to(device)
        
        # Additional device movement to ensure everything is on CUDA
        printt("Performing comprehensive device movement...")
        def move_all_to_device(module, target_device):
            """Recursively move all components to target device"""
            module.to(target_device)
            for name, child in module.named_children():
                move_all_to_device(child, target_device)
            for name, param in module.named_parameters():
                if param.device != target_device:
                    param.data = param.data.to(target_device)
                    printt(f"Moved parameter {name} to {target_device}")
        
        if device == 'cuda':
            move_all_to_device(supir_model, device)
        
        # The model uses mixed precision by design (bf16 for AE, fp16 for diffusion)
        # This is intentional and should not be changed
        printt(f"Model loaded with mixed precision: ae_dtype={supir_model.ae_dtype}, diffusion_dtype={supir_model.model.dtype}")
        
        printt(f"SUPIR model loaded successfully")
        return True
        
    except Exception as e:
        printt(f"Error loading SUPIR model: {str(e)}")
        return False

def load_llava_model():
    global llava_agent
    
    try:
        printt("Loading LLaVA model")
        
        # LLaVA model path
        llava_model_path = os.path.join(MODELS_DIR, "llava-v1.5-7b")
        
        if not os.path.exists(llava_model_path):
            printt(f"LLaVA model not found at: {llava_model_path}")
            return False
        
        printt(f"Loading LLaVA from: {llava_model_path}")
        llava_agent = LLavaAgent(
            model_path=llava_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            conv_mode='vicuna_v1'
        )
        printt("LLaVA model loaded successfully")
        return True
    except Exception as e:
        printt(f"Error loading LLaVA model: {str(e)}")
        printt(traceback.format_exc())
        return False

# Job processing functions
async def process_job(job_id: str, image_path: str, settings: JobSettings):
    """Process a single image job"""
    global supir_model, llava_agent
    
    try:
        # Update job status
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["progress"] = 0.1
        jobs[job_id]["message"] = "Starting processing..."
        
        # Load models if needed
        if supir_model is None and settings.apply_supir:
            if not load_supir_model(settings.model, settings.checkpoint_type):
                raise Exception("Failed to load SUPIR model")
        
        if llava_agent is None and settings.apply_llava:
            if not load_llava_model():
                raise Exception("Failed to load LLaVA model")
        
        # Load and prepare image
        jobs[job_id]["progress"] = 0.2
        jobs[job_id]["message"] = "Loading image..."
        
        input_image = Image.open(image_path).convert('RGB')
        
        # Create MediaData object
        media_data = MediaData(
            media_path=image_path,
            media_type="image"
        )
        media_data.media_data = np.array(input_image)
        
        # Apply LLaVA if enabled
        caption = ""
        if settings.apply_llava and llava_agent:
            jobs[job_id]["progress"] = 0.3
            jobs[job_id]["message"] = "Generating caption with LLaVA..."
            
            try:
                # Convert PIL image to format expected by LLaVA
                lq = HWC3(np.array(input_image))
                lq_pil = Image.fromarray(lq.astype('uint8'))
                caption = llava_agent.gen_image_caption([lq_pil], temperature=0.2, top_p=0.7)[0]
                media_data.caption = caption
                
                if settings.save_captions:
                    caption_path = os.path.splitext(image_path)[0] + "_caption.txt"
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
            except Exception as e:
                printt(f"LLaVA processing failed: {str(e)}")
                caption = ""
        
        # Apply SUPIR if enabled
        if settings.apply_supir and supir_model:
            jobs[job_id]["progress"] = 0.4
            jobs[job_id]["message"] = "Processing with SUPIR..."
            
            # Prepare image tensor
            img_array = np.array(input_image)
            img_array = HWC3(img_array)
            
            # Upscale image
            if settings.upscale_size > 1:
                img_array = upscale_image(img_array, settings.upscale_size)
            
            # Convert to tensor
            printt(f"Converting image to tensor, img_array shape: {img_array.shape}")
            pil_image = Image.fromarray(img_array)
            printt(f"PIL image size: {pil_image.size}")
            
            tensor_result = PIL2Tensor(pil_image, upscale=1)
            printt(f"PIL2Tensor returned: {type(tensor_result)}, length: {len(tensor_result) if hasattr(tensor_result, '__len__') else 'N/A'}")
            
            img_tensor, h0, w0 = tensor_result
            img_tensor = img_tensor.unsqueeze(0)
            printt(f"Tensor shape: {img_tensor.shape}, h0: {h0}, w0: {w0}")
            
            # Move to device and handle mixed precision properly
            device = next(supir_model.parameters()).device
            model_dtype = next(supir_model.parameters()).dtype
            printt(f"Model device: {device}, Model dtype: {model_dtype}")
            printt(f"Input tensor dtype before conversion: {img_tensor.dtype}")
            printt(f"Model ae_dtype: {supir_model.ae_dtype}")
            
            # Convert input tensor to match the model's expected input type
            # The model expects float32 input tensors for proper mixed precision handling
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            printt(f"Input tensor dtype after conversion: {img_tensor.dtype}")
            printt(f"Input tensor device: {img_tensor.device}")
            
            # Double-check that critical model components are on the right device
            if hasattr(supir_model, 'first_stage_model') and hasattr(supir_model.first_stage_model, 'quant_conv'):
                quant_conv_device = next(supir_model.first_stage_model.quant_conv.parameters()).device
                printt(f"quant_conv device: {quant_conv_device}")
                if quant_conv_device != device:
                    printt(f"WARNING: quant_conv is on {quant_conv_device}, moving to {device}")
                    supir_model.first_stage_model.quant_conv.to(device)
            
            # Prepare prompt
            if settings.prompt:
                prompt = settings.prompt
            elif caption:
                prompt = caption
            else:
                prompt = "high quality, detailed"
            
            # Add style prompt
            if settings.prompt_style == "Photorealistic":
                prompt = f"photorealistic, {prompt}"
            
            jobs[job_id]["progress"] = 0.6
            jobs[job_id]["message"] = "Running SUPIR inference..."
            
            # Run SUPIR inference
            with torch.no_grad():
                result = supir_model.batchify_sample(
                    x=img_tensor,
                    p=[prompt],
                    num_steps=settings.edm_steps,
                    restoration_scale=settings.s_stage1,
                    s_churn=settings.s_churn,
                    s_noise=settings.s_noise,
                    cfg_scale=settings.s_cfg,
                    seed=settings.seed,
                    color_fix_type=settings.color_fix_type,
                    use_linear_cfg=settings.linear_cfg
                )
            
            if result is not None:
                # Debug: Check tensor values before conversion
                printt(f"Result tensor shape: {result[0].shape}")
                printt(f"Result tensor dtype: {result[0].dtype}")
                printt(f"Result tensor device: {result[0].device}")
                printt(f"Result tensor min: {result[0].min().item()}")
                printt(f"Result tensor max: {result[0].max().item()}")
                printt(f"Result tensor mean: {result[0].mean().item()}")
                printt(f"Result tensor std: {result[0].std().item()}")
                
                # Get the result tensor
                result_tensor = result[0].clone()
                
                # Check for NaN or Inf values
                if torch.isnan(result_tensor).any():
                    printt("WARNING: NaN values detected in result tensor!")
                    result_tensor = torch.nan_to_num(result_tensor, nan=0.0)
                
                if torch.isinf(result_tensor).any():
                    printt("WARNING: Inf values detected in result tensor!")
                    result_tensor = torch.nan_to_num(result_tensor, posinf=1.0, neginf=-1.0)
                
                # The SUPIR VAE decoder should output values in [-1, 1] range
                # But sometimes it might output in a different range due to precision issues
                tensor_min = result_tensor.min().item()
                tensor_max = result_tensor.max().item()
                
                # If the tensor is clearly outside the expected range, normalize it
                if tensor_min < -2.0 or tensor_max > 2.0:
                    printt(f"Tensor values are outside expected range [{tensor_min}, {tensor_max}], normalizing...")
                    # Normalize to [-1, 1] range
                    result_tensor = 2.0 * (result_tensor - tensor_min) / (tensor_max - tensor_min) - 1.0
                elif tensor_min >= 0 and tensor_max <= 1.0:
                    printt("Tensor appears to be in [0, 1] range, converting to [-1, 1]")
                    result_tensor = result_tensor * 2.0 - 1.0
                elif tensor_min >= 0 and tensor_max > 1.0 and tensor_max <= 255.0:
                    printt("Tensor appears to be in [0, 255] range, converting to [-1, 1]")
                    result_tensor = (result_tensor / 255.0) * 2.0 - 1.0
                else:
                    # Clamp to [-1, 1] range to be safe
                    printt(f"Tensor range [{tensor_min}, {tensor_max}], clamping to [-1, 1]")
                    result_tensor = torch.clamp(result_tensor, -1.0, 1.0)
                
                printt(f"Final tensor min: {result_tensor.min().item()}")
                printt(f"Final tensor max: {result_tensor.max().item()}")
                
                # Convert back to PIL
                result_image = Tensor2PIL(result_tensor, h0, w0)
                
                # Save result
                output_path = os.path.join(BATCH_OUTPUT_DIR, f"{job_id}_result.png")
                result_image.save(output_path, "PNG")
                
                jobs[job_id]["result_path"] = output_path
            else:
                raise Exception("SUPIR processing returned None")
        else:
            # If SUPIR is disabled, just copy the original image
            output_path = os.path.join(BATCH_OUTPUT_DIR, f"{job_id}_result.png")
            input_image.save(output_path, "PNG")
            jobs[job_id]["result_path"] = output_path
        
        # Complete job
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["message"] = "Processing completed successfully"
        jobs[job_id]["completed_at"] = datetime.now()
        
        printt(f"Job {job_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Job processing failed: {str(e)}"
        printt(error_msg)
        printt(traceback.format_exc())
        
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = error_msg
        jobs[job_id]["completed_at"] = datetime.now()
    
    finally:
        # Cleanup temporary files
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

# API Endpoints
@app.post("/job", response_model=JobResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    settings: str = Form(default="{}"),
    token: str = Depends(verify_token)
):
    """Create a new image processing job"""
    
    # Validate image file
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Parse settings
    try:
        settings_dict = json.loads(settings) if settings else {}
        job_settings = JobSettings(**settings_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid settings: {str(e)}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded image
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, f"{job_id}_{image.filename}")
    
    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")
    
    # Create job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.now(),
        "progress": 0.0,
        "message": "Job created, waiting to start...",
        "settings": job_settings,
        "image_path": image_path
    }
    
    # Start background processing
    background_tasks.add_task(process_job, job_id, image_path, job_settings)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=jobs[job_id]["created_at"],
        message="Job created successfully"
    )

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, token: str = Depends(verify_token)):
    """Get the status of a specific job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        progress=job.get("progress", 0.0),
        message=job.get("message", ""),
        error=job.get("error")
    )

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str, token: str = Depends(verify_token)):
    """Retrieve the final processed image"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result_path = job.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result_path,
        media_type="image/png",
        filename=f"{job_id}_result.png"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models"""
    models = []
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith(('.safetensors', '.ckpt', '.pth')):
                models.append(file)
    return {"models": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 