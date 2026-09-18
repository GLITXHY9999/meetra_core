from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

with open('employee_data.csv', 'rb') as f:
    data = f.read()

resp = client.post('/api/upload-and-train', data={'target_column': 'Attrition', 'task_type': 'classification'}, files={'file': ('employee_data.csv', data)})
print(f"High risk length: {len(resp.json()['result']['high_risk_employees'])}")
