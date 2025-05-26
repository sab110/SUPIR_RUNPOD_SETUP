# SUPIR REST API

A REST API for SUPIR (Scaling-UP Image Restoration) that provides image upscaling and enhancement capabilities through HTTP endpoints.

## Features

- **REST API Endpoints**: Three main endpoints for job creation, status checking, and result retrieval
- **Authentication**: Bearer token authentication for secure access
- **Asynchronous Processing**: Background job processing with progress tracking
- **LLaVA Integration**: Automatic caption generation for enhanced results
- **Flexible Settings**: Comprehensive configuration options for processing
- **RunPod Compatible**: Optimized for RunPod deployment with network storage support
- **Docker Support**: Containerized deployment with GPU support

## API Endpoints

### POST /job
Create a new image processing job.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Headers: `Authorization: Bearer <token>`
- Body:
  - `image`: Image file (required)
  - `settings`: JSON string with processing settings (optional)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "created_at": "2024-01-01T12:00:00",
  "message": "Job created successfully"
}
```

### GET /job/:id
Get the status of a specific job.

**Request:**
- Method: `GET`
- Headers: `Authorization: Bearer <token>`

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "created_at": "2024-01-01T12:00:00",
  "completed_at": null,
  "progress": 0.6,
  "message": "Running SUPIR inference...",
  "error": null
}
```

### GET /job/:id/result
Download the processed image.

**Request:**
- Method: `GET`
- Headers: `Authorization: Bearer <token>`

**Response:**
- Content-Type: `image/png`
- Body: Processed image file

## Settings Configuration

All settings are optional and have sensible defaults:

```json
{
  "upscale_size": 2,
  "apply_llava": true,
  "apply_supir": true,
  "prompt_style": "Photorealistic",
  "model": "RealVisXL_V5.0_fp16.safetensors",
  "checkpoint_type": "Standard SDXL",
  "prompt": "",
  "save_captions": true,
  "text_guidance_scale": 1024,
  "background_restoration": true,
  "face_restoration": true,
  "edm_steps": 50,
  "s_stage1": 1.0,
  "s_stage2": 1.0,
  "s_cfg": 7.5,
  "seed": -1,
  "sampler": "DPMPP2M",
  "s_churn": 0,
  "s_noise": 1.003,
  "color_fix_type": "Wavelet",
  "linear_cfg": false,
  "linear_s_stage2": false
}
```

## Environment Variables

Configure the API using these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOADING_MODE` | `FP16` | Model precision (FP16/FP32/BF16) |
| `TILED_VAE` | `True` | Enable tiled VAE for memory efficiency |
| `AUTO_MOVE_CPU` | `True` | Automatically move models to CPU when not in use |
| `API_TOKEN` | `your-secret-token-here` | Authentication token |
| `WORKSPACE_DIR` | `/workspace` | Main workspace directory |
| `MODELS_DIR` | `/workspace/models` | Directory containing model files |

## Installation & Setup

### Option 1: Docker (Recommended)

1. **Build the Docker image:**
```bash
docker build -t supir-api .
```

2. **Run with docker-compose:**
```bash
# Edit docker-compose.yml to set your API token
docker-compose up -d
```

3. **Or run directly:**
```bash
docker run -d \
  --name supir-api \
  --gpus all \
  -p 8000:8000 \
  -e API_TOKEN=your-secret-token-here \
  -v ./models:/workspace/models \
  -v ./outputs:/workspace/adjustedupscaled \
  supir-api
```

### Option 2: Local Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

2. **Set environment variables:**
```bash
export API_TOKEN=your-secret-token-here
export LOADING_MODE=FP16
export TILED_VAE=True
export AUTO_MOVE_CPU=True
```

3. **Run the API server:**
```bash
python api_server.py
```

## RunPod Deployment

This API is optimized for RunPod deployment with network storage:

1. **Build and push your Docker image:**
```bash
docker build -t your-registry/supir-api .
docker push your-registry/supir-api
```

2. **Create a RunPod template:**
   - Use your Docker image
   - Set environment variables (especially `API_TOKEN`)
   - Mount network storage to `/workspace` (the entire workspace directory)
   - Expose port 8000

3. **Network Storage Setup:**
   - Upload your SUPIR models to RunPod network storage at `/workspace/models`
   - The entire workspace directory is network-mounted, so models and outputs persist across runs
   - Place your model files (e.g., `RealVisXL_V5.0_fp16.safetensors`) directly in `/workspace/models/`
   - Processed images will be saved to `/workspace/adjustedupscaled/`

4. **Required Models:**
   - Download and place the following in `/workspace/models/`:
     - `RealVisXL_V5.0_fp16.safetensors` (default model)
     - Any other SUPIR-compatible models you want to use
   - LLaVA models will be downloaded automatically on first use

## Usage Examples

### Python Client

Use the provided `api_client_example.py`:

```python
from api_client_example import SUPIRClient

client = SUPIRClient(
    base_url="http://localhost:8000",
    token="your-secret-token-here"
)

# Process an image
success = client.process_image(
    image_path="input.jpg",
    output_path="output.png",
    settings={
        "upscale_size": 4,
        "prompt": "high quality, detailed",
        "edm_steps": 100
    }
)
```

### cURL Examples

**Create a job:**
```bash
curl -X POST "http://localhost:8000/job" \
  -H "Authorization: Bearer your-secret-token-here" \
  -F "image=@input.jpg" \
  -F 'settings={"upscale_size": 2, "prompt": "high quality"}'
```

**Check job status:**
```bash
curl -X GET "http://localhost:8000/job/job-id-here" \
  -H "Authorization: Bearer your-secret-token-here"
```

**Download result:**
```bash
curl -X GET "http://localhost:8000/job/job-id-here/result" \
  -H "Authorization: Bearer your-secret-token-here" \
  -o result.png
```

## Model Requirements

Place your SUPIR model files in the models directory:

- **SUPIR Models**: `v0F.ckpt`, `v0Q.ckpt`
- **SDXL Checkpoints**: `.safetensors` or `.ckpt` files
- **LLaVA Models**: Automatically downloaded on first use

Example model structure:
```
models/
├── RealVisXL_V5.0_fp16.safetensors
├── v0F.ckpt
├── v0Q.ckpt
└── other_checkpoints.safetensors
```

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## Health Monitoring

The API includes health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# List available models
curl -H "Authorization: Bearer your-token" http://localhost:8000/models
```

## Performance Optimization

### Memory Management
- **Tiled VAE**: Enabled by default for memory efficiency
- **Auto CPU Move**: Models automatically moved to CPU when idle
- **FP16 Precision**: Reduces memory usage while maintaining quality

### GPU Optimization
- **CUDA 11.8**: Optimized for modern GPUs
- **Batch Processing**: Efficient handling of multiple requests
- **Memory Cleanup**: Automatic garbage collection between jobs

## Troubleshooting

### Common Issues

1. **Out of Memory Errors:**
   - Reduce `upscale_size`
   - Enable `TILED_VAE=True`
   - Use `LOADING_MODE=FP16`

2. **Model Loading Failures:**
   - Check model file paths
   - Verify file permissions
   - Ensure sufficient disk space

3. **Authentication Errors:**
   - Verify `API_TOKEN` environment variable
   - Check Authorization header format

### Logs

Monitor logs for debugging:
```bash
# Docker logs
docker logs supir-api

# Direct execution
python api_server.py
```

## Security Considerations

- **Authentication**: Always use a strong API token
- **Network**: Run behind a reverse proxy in production
- **File Upload**: Validate image files and limit sizes
- **Rate Limiting**: Consider implementing rate limiting for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project follows the same license as the original SUPIR project.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub 