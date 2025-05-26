#!/usr/bin/env python3
"""
Test script for SUPIR API

This script tests the basic functionality of the SUPIR REST API.
"""

import requests
import json
import time
import os
import tempfile
from PIL import Image
import numpy as np

def create_test_image(width=512, height=512):
    """Create a simple test image"""
    # Create a simple gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient pattern
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(255 * x / width),      # Red gradient
                int(255 * y / height),     # Green gradient
                128                        # Blue constant
            ]
    
    return Image.fromarray(img_array)

def test_api():
    """Test the SUPIR API endpoints"""
    base_url = "http://localhost:8000"
    token = "your-secret-token-here"
    headers = {"Authorization": f"Bearer {token}"}
    
    print("🧪 Testing SUPIR API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: List models (requires auth)
    print("\n2. Testing model listing...")
    try:
        response = requests.get(f"{base_url}/models", headers=headers)
        if response.status_code == 200:
            models = response.json()
            print("✅ Model listing passed")
            print(f"   Available models: {models['models']}")
        else:
            print(f"❌ Model listing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Model listing failed: {e}")
    
    # Test 3: Authentication test (invalid token)
    print("\n3. Testing authentication...")
    try:
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        response = requests.get(f"{base_url}/models", headers=invalid_headers)
        if response.status_code == 401:
            print("✅ Authentication test passed (correctly rejected invalid token)")
        else:
            print(f"❌ Authentication test failed: expected 401, got {response.status_code}")
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
    
    # Test 4: Create and process a job
    print("\n4. Testing job creation and processing...")
    
    # Create a test image
    test_image = create_test_image(256, 256)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_image.save(tmp_file.name, 'PNG')
        test_image_path = tmp_file.name
    
    try:
        # Create job with minimal settings
        settings = {
            "upscale_size": 2,
            "apply_llava": False,  # Disable LLaVA for faster testing
            "apply_supir": True,
            "edm_steps": 10,       # Reduce steps for faster testing
            "prompt": "test image"
        }
        
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            data = {'settings': json.dumps(settings)}
            
            response = requests.post(
                f"{base_url}/job",
                files=files,
                data=data,
                headers=headers
            )
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data['job_id']
            print(f"✅ Job created successfully")
            print(f"   Job ID: {job_id}")
            
            # Test 5: Monitor job status
            print("\n5. Testing job status monitoring...")
            max_wait = 300  # 5 minutes max
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_response = requests.get(f"{base_url}/job/{job_id}", headers=headers)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data['status']
                    progress = status_data.get('progress', 0.0)
                    message = status_data.get('message', '')
                    
                    print(f"   Status: {status} | Progress: {progress:.1%} | {message}")
                    
                    if status == 'completed':
                        print("✅ Job completed successfully")
                        
                        # Test 6: Download result
                        print("\n6. Testing result download...")
                        result_response = requests.get(f"{base_url}/job/{job_id}/result", headers=headers)
                        
                        if result_response.status_code == 200:
                            # Save result to file
                            result_path = f"test_result_{job_id}.png"
                            with open(result_path, 'wb') as f:
                                f.write(result_response.content)
                            print(f"✅ Result downloaded successfully: {result_path}")
                            
                            # Verify it's a valid image
                            try:
                                result_image = Image.open(result_path)
                                print(f"   Result image size: {result_image.size}")
                                print(f"   Result image mode: {result_image.mode}")
                            except Exception as e:
                                print(f"❌ Result image validation failed: {e}")
                        else:
                            print(f"❌ Result download failed: {result_response.status_code}")
                        
                        break
                    elif status == 'failed':
                        error = status_data.get('error', 'Unknown error')
                        print(f"❌ Job failed: {error}")
                        break
                    
                    time.sleep(5)  # Wait 5 seconds before checking again
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    break
            else:
                print(f"❌ Job timed out after {max_wait} seconds")
        
        else:
            print(f"❌ Job creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Job processing test failed: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)
    
    print("\n" + "=" * 50)
    print("🏁 API testing completed")

if __name__ == "__main__":
    test_api() 