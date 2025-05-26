import asyncio
import os
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import traceback
from pathlib import Path

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
LOADING_MODE = os.getenv("LOADING_MODE", "FP16")
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
    model: str = Field(default="RealVisXL_V5.0_fp16.safetensors", description="Model checkpoint")
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
        
        # Model checkpoint path
        ckpt_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        supir_model = create_SUPIR_model(
            config_path=config_path,
            weight_dtype=weight_dtype,
            device=device,
            ckpt=ckpt_path
        )
        
        # Initialize tiled VAE if enabled
        if TILED_VAE:
            supir_model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
        
        # Move to device
        supir_model.move_to(device)
        
        printt(f"SUPIR model loaded successfully")
        return True
        
    except Exception as e:
        printt(f"Error loading SUPIR model: {str(e)}")
        return False

def load_llava_model():
    global llava_agent
    
    try:
        printt("Loading LLaVA model")
        llava_agent = LLavaAgent()
        printt("LLaVA model loaded successfully")
        return True
    except Exception as e:
        printt(f"Error loading LLaVA model: {str(e)}")
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
            img_tensor, h0, w0 = PIL2Tensor(Image.fromarray(img_array), upscale=1)
            img_tensor = img_tensor.unsqueeze(0)
            
            # Move to device
            device = next(supir_model.parameters()).device
            img_tensor = img_tensor.to(device)
            
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
                # Convert back to PIL
                result_image = Tensor2PIL(result[0], h0, w0)
                
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