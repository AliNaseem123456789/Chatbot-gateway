import requests
from PIL import Image
import io

# Load your WebP image

webp_path = r"E:/Work/Crestmartllc/website/ecommerce-app/public/a.jpg"
img = Image.open(webp_path)
import requests
import base64
from PIL import Image
import io

# Load and heavily compress your image
webp_path = r"E:/Work/Crestmartllc/website/ecommerce-app/public/a.jpg"
img = Image.open(webp_path)

# Aggressively resize and compress
img.thumbnail((300, 300))  # Much smaller
buffer = io.BytesIO()
img.convert('RGB').save(buffer, format='JPEG', quality=40)  # Lower quality
base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

print(f"Compressed size: {len(buffer.getvalue()) / 1024:.1f} KB")

url = "https://chatbot-gateway-production-208c.up.railway.app/api/chat/ecommerce/image/base64"
data = {
    "image_base64": base64_image,
    "message": "What is this product?",
    "user_id": "test123"
}

try:
    response = requests.post(url, json=data, timeout=60)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.text}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
# Convert to JPEG
jpeg_buffer = io.BytesIO()
img.convert('RGB').save(jpeg_buffer, format='JPEG', quality=85)
jpeg_buffer.seek(0)

url = "https://chatbot-gateway.onrender.com/api/chat/ecommerce/image"
files = {"image": ("OIP.jpg", jpeg_buffer, "image/jpeg")}
data = {"message": "What is this product?", "user_id": "test123"}

response = requests.post(url, files=files, data=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")