#!/usr/bin/env python
# coding: utf-8

import requests
import base64


service_name = 'intel'
host = f'{service_name}.default.example.com'

actual_domain = 'http://localhost:8080'
model_name='cifar34'
service_url = f'{actual_domain}/v1/models/{model_name}:predict'

headers = {'Host': host}


# request = {
#     "instances": [
#         'https://github.com/mmgxa/E2_7/raw/main/153.jpg',
#         'https://github.com/mmgxa/E2_7/raw/main/153.jpg'
#     ]
# }

image = open('153.jpg', 'rb') #open binary file in read mode
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
bytes_array = image_64_encode.decode('utf-8')
request = {
  "instances":[
    {
      "data": bytes_array
    }
  ]
}


response = requests.post(service_url, json=request, headers=headers)

# print(response)
# print(response.content)
print(response.json())

