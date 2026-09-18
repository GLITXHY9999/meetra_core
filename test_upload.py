import urllib.request
import urllib.parse
import json

with open('employee_data.csv', 'rb') as f:
    data = f.read()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    b'--' + boundary.encode() + b'\r\n'
    b'Content-Disposition: form-data; name="target_column"\r\n\r\n'
    b'Attrition\r\n'
    b'--' + boundary.encode() + b'\r\n'
    b'Content-Disposition: form-data; name="task_type"\r\n\r\n'
    b'classification\r\n'
    b'--' + boundary.encode() + b'\r\n'
    b'Content-Disposition: form-data; name="file"; filename="employee_data.csv"\r\n'
    b'Content-Type: text/csv\r\n\r\n'
    + data + b'\r\n'
    b'--' + boundary.encode() + b'--\r\n'
)

req = urllib.request.Request(
    'http://localhost:8000/api/upload-and-train',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print(f'HTTP Error: {e.code}')
    print(e.read().decode())
