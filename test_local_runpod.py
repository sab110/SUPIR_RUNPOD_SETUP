#!/usr/bin/env python3
"""
Local RunPod Test Script for SUPIR API

This script tests the SUPIR API running locally on RunPod.
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

def test_runpod_api():
    """Test the SUPIR API on RunPod"""
    # RunPod typically exposes services on localhost
    base_url = "http://localhost:8000"
    token = os.getenv("API_TOKEN", "your-secure-token-here")
    headers = {"Authorization": f"Bearer {token}"}
    
    print("🧪 Testing SUPIR API on RunPod")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print(f"Token: {token[:10]}..." if len(token) > 10 else f"Token: {token}")
    print()
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the server running?")
        print("   Try: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: List models
    print("\n2. Testing model listing...")
    try:
        response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✅ Model listing passed")
            print(f"   Available models: {models['models']}")
            if not models['models']:
                print("   ⚠️  No models found - upload models to /workspace/models/")
        elif response.status_code == 401:
            print("❌ Authentication failed - check your API_TOKEN")
            return False
        else:
            print(f"❌ Model listing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Model listing failed: {e}")
    
    # Test 3: Create a simple job
    print("\n3. Testing job creation...")
    
    # Create a test image
    test_image = create_test_image(256, 256)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_image.save(tmp_file.name, 'PNG')
        test_image_path = tmp_file.name
    
    try:
        # Create job with fast settings for testing
        settings = {
            "upscale_size": 2,
            "apply_llava": False,  # Disable for faster testing
            "apply_supir": True,
            "edm_steps": 5,        # Very low for quick test
            "prompt": "test image"
        }
        
        print(f"   Creating job with settings: {settings}")
        
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            data = {'settings': json.dumps(settings)}
            
            response = requests.post(
                f"{base_url}/job",
                files=files,
                data=data,
                headers=headers,
                timeout=30
            )
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data['job_id']
            print(f"✅ Job created successfully")
            print(f"   Job ID: {job_id}")
            
            # Test 4: Monitor job
            print("\n4. Monitoring job progress...")
            max_wait = 120  # 2 minutes max for test
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    status_response = requests.get(f"{base_url}/job/{job_id}", headers=headers, timeout=10)
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data['status']
                        progress = status_data.get('progress', 0.0)
                        message = status_data.get('message', '')
                        
                        print(f"   Status: {status} | Progress: {progress:.1%} | {message}")
                        
                        if status == 'completed':
                            print("✅ Job completed successfully")
                            
                            # Test 5: Download result
                            print("\n5. Testing result download...")
                            try:
                                result_response = requests.get(f"{base_url}/job/{job_id}/result", headers=headers, timeout=30)
                                
                                if result_response.status_code == 200:
                                    result_path = f"test_result_{job_id}.png"
                                    with open(result_path, 'wb') as f:
                                        f.write(result_response.content)
                                    print(f"✅ Result downloaded: {result_path}")
                                    
                                    # Verify image
                                    try:
                                        result_image = Image.open(result_path)
                                        print(f"   Result size: {result_image.size}")
                                        print(f"   Original size: {test_image.size}")
                                    except Exception as e:
                                        print(f"❌ Result validation failed: {e}")
                                else:
                                    print(f"❌ Download failed: {result_response.status_code}")
                            except Exception as e:
                                print(f"❌ Download failed: {e}")
                            
                            break
                        elif status == 'failed':
                            error = status_data.get('error', 'Unknown error')
                            print(f"❌ Job failed: {error}")
                            break
                        
                        time.sleep(3)  # Check every 3 seconds
                    else:
                        print(f"❌ Status check failed: {status_response.status_code}")
                        break
                except Exception as e:
                    print(f"❌ Status check error: {e}")
                    break
            else:
                print(f"❌ Job timed out after {max_wait} seconds")
        
        else:
            print(f"❌ Job creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Job test failed: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)
    
    print("\n" + "=" * 50)
    print("🏁 RunPod API testing completed")
    print("\n📋 Next steps:")
    print("   1. Upload your SUPIR models to /workspace/models/")
    print("   2. Test with real images using the API client")
    print("   3. Monitor logs for any issues")

if __name__ == "__main__":
    test_runpod_api() 