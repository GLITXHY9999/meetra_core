import json
import os

transcript_path = r"C:\Users\HP\.gemini\antigravity\brain\41f98e32-df2b-41c1-8c93-4e4b4d16c651\.system_generated\logs\transcript_full.jsonl"
if not os.path.exists(transcript_path):
    transcript_path = transcript_path.replace("_full", "")

largest_csv = ""
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            content = data.get('content', '')
            if 'Age,Attrition,BusinessTravel' in content:
                csv_start = content.find('Age,Attrition,BusinessTravel')
                csv_candidate = content[csv_start:]
                csv_candidate = csv_candidate.split('</USER_REQUEST>')[0].strip()
                if len(csv_candidate) > len(largest_csv):
                    largest_csv = csv_candidate
        except Exception as e:
            pass

if largest_csv:
    with open(r'c:\Users\HP\Desktop\meetra_core\employee_data.csv', 'w', encoding='utf-8') as out_f:
        out_f.write(largest_csv)
    print("Extracted", len(largest_csv), "bytes to employee_data.csv")
else:
    print("Could not find the dataset!")
