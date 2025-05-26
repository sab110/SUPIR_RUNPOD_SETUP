#!/usr/bin/env python3
"""
SUPIR API Client Example

This script demonstrates how to use the SUPIR REST API to process images.
"""

import requests
import json
import time
import os
from typing import Optional

class SUPIRClient:
    def __init__(self, base_url: str = "http://localhost:8000", token: str = "your-secret-token-here"):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def health_check(self) -> dict:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> dict:
        """List available models"""
        response = requests.get(f"{self.base_url}/models", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def create_job(self, image_path: str, settings: Optional[dict] = None) -> dict:
        """Create a new processing job"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Default settings
        default_settings = {
            "upscale_size": 2,
            "apply_llava": True,
            "apply_supir": True,
            "prompt_style": "Photorealistic",
            "model": "RealVisXL_V5.0_fp16.safetensors",
            "checkpoint_type": "Standard SDXL",
            "prompt": "",
            "save_captions": True,
            "text_guidance_scale": 1024,
            "background_restoration": True,
            "face_restoration": True,
            "edm_steps": 50,
            "s_stage1": 1.0,
            "s_stage2": 1.0,
            "s_cfg": 7.5,
            "seed": -1,
            "sampler": "DPMPP2M",
            "s_churn": 0,
            "s_noise": 1.003,
            "color_fix_type": "Wavelet",
            "linear_cfg": False,
            "linear_s_stage2": False
        }
        
        if settings:
            default_settings.update(settings)
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'settings': json.dumps(default_settings)
            }
            
            response = requests.post(
                f"{self.base_url}/job",
                files=files,
                data=data,
                headers=self.headers
            )
        
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> dict:
        """Get job status"""
        response = requests.get(f"{self.base_url}/job/{job_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def download_result(self, job_id: str, output_path: str) -> bool:
        """Download the processed image"""
        response = requests.get(f"{self.base_url}/job/{job_id}/result", headers=self.headers)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
    
    def process_image(self, image_path: str, output_path: str, settings: Optional[dict] = None, 
                     poll_interval: int = 5, timeout: int = 300) -> bool:
        """
        Complete workflow: create job, wait for completion, download result
        
        Args:
            image_path: Path to input image
            output_path: Path to save processed image
            settings: Processing settings (optional)
            poll_interval: How often to check job status (seconds)
            timeout: Maximum time to wait (seconds)
        
        Returns:
            True if successful, False otherwise
        """
        print(f"Creating job for image: {image_path}")
        job_response = self.create_job(image_path, settings)
        job_id = job_response['job_id']
        print(f"Job created with ID: {job_id}")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_response = self.get_job_status(job_id)
            status = status_response['status']
            progress = status_response.get('progress', 0.0)
            message = status_response.get('message', '')
            
            print(f"Status: {status} | Progress: {progress:.1%} | {message}")
            
            if status == 'completed':
                print("Job completed successfully!")
                self.download_result(job_id, output_path)
                print(f"Result saved to: {output_path}")
                return True
            elif status == 'failed':
                error = status_response.get('error', 'Unknown error')
                print(f"Job failed: {error}")
                return False
            
            time.sleep(poll_interval)
        
        print(f"Timeout reached after {timeout} seconds")
        return False


def main():
    """Example usage"""
    # Initialize client
    client = SUPIRClient(
        base_url="http://localhost:8000",
        token="your-secret-token-here"
    )
    
    try:
        # Health check
        print("Checking API health...")
        health = client.health_check()
        print(f"API Status: {health['status']}")
        
        # List available models
        print("\nListing available models...")
        models = client.list_models()
        print(f"Available models: {models['models']}")
        
        # Example image processing
        image_path = "example_image.jpg"  # Replace with your image path
        output_path = "processed_image.png"
        
        if os.path.exists(image_path):
            # Custom settings example
            custom_settings = {
                "upscale_size": 4,
                "prompt": "high quality, detailed, sharp",
                "edm_steps": 100,
                "s_cfg": 8.0
            }
            
            print(f"\nProcessing image: {image_path}")
            success = client.process_image(
                image_path=image_path,
                output_path=output_path,
                settings=custom_settings
            )
            
            if success:
                print("Image processing completed successfully!")
            else:
                print("Image processing failed!")
        else:
            print(f"Example image not found: {image_path}")
            print("Please provide a valid image path to test the API.")
    
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 