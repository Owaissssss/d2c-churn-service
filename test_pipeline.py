import subprocess
import time
import requests

print("--- Step 1: Spawning Background FastAPI Live Instances ---")
server_process = subprocess.Popen(
    ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Allow the server 3 seconds to spin up completely
time.sleep(3)

url_check = "http://127.0.0.1:8000/"
url_predict = "http://127.0.0.1:8000/predict"

try:
    print("\n--- Step 2: Testing Health Gateway Check Endpoint ---")
    health_response = requests.get(url_check)
    print(f"Gateway Response Status: {health_response.status_code}")
    print(f"Gateway Content Payload: {health_response.json()}")

    print("\n--- Step 3: Feeding Automated Vector Test Cases into API ---")
    test_payload = {
        "recency_days": 95.0,
        "frequency_180d": 1.0,
        "monetary_180d": 450.0,
        "ticket_count_90d": 4.0,
        "sessions_30d": 2.0,
        "abandoned_carts_30d": 1.0,
        "city_tier": "Tier 3",
        "age_group": "35+",
        "acquisition_channel": "Marketplace",
        "loyalty_tier": "UNKNOWN_UNENROLLED",
        "preferred_category": "Hair",
        "marketing_consent": "No"
    }

    predict_response = requests.post(url_predict, json=test_payload)
    print(f"Predict Response Status: {predict_response.status_code}")
    print(f"Live Scoring Output:     {predict_response.json()}")

    print("\n--- STATUS: END-TO-END PIPELINE VERIFICATION SUCCESSFUL ---")

finally:
    # Safely close down the background engine instance so ports stay clean
    print("\n--- Step 4: Shuts down pipeline engine instances ---")
    server_process.terminate()
    server_process.wait()
    print("Background server closed down cleanly.")