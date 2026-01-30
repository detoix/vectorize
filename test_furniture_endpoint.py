import requests
import os
import json

def test_furniture_endpoint():
    url = "http://localhost:8010/furniture"
    image_path = "/home/kdeptula/.gemini/antigravity/brain/f75aed76-291f-4d95-b7c4-410eb66a8f01/uploaded_media_1769759276928.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Testing endpoint {url} with image {image_path}")
    
    with open(image_path, "rb") as img:
        files = {"image": ("test_image.png", img, "image/png")}
        try:
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response JSON:")
                print(json.dumps(response.json(), indent=2))
            else:
                print("Response Text:")
                print(response.text)
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to server. Is it running?")

if __name__ == "__main__":
    test_furniture_endpoint()
