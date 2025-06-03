import requests
url = 'http://localhost:5000/predict'
image_path = 'test_image.jpg'
with open(image_path, 'rb') as img:
    response = requests.post(url, files={'image': img})

print(response.json())
