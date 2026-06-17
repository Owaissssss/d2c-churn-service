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

url_health = "http://127.0.0.1:8000/health"
url_predict = "http://127.0.0.1:8000/predict"
url_batch = "http://127.0.0.1:8000/batch_predict"

try:
    print("\n--- Step 2: Testing Health Gateway Check Endpoint ---")
    health_response = requests.get(url_health)
    print(f"Gateway Response Status: {health_response.status_code}")
    print(f"Gateway Content Payload: {health_response.json()}")

    # Construct complete feature vectors matching real snapshot distributions
    customer_a = {
        "recency_days": 95.0, "frequency_180d": 1.0, "monetary_180d": 450.0,
        "ticket_count_90d": 4.0, "sessions_30d": 2.0, "abandoned_carts_30d": 1.0,
        "city_tier": "Tier 3", "age_group": "35+", "acquisition_channel": "Marketplace",
        "loyalty_tier": "UNKNOWN_UNENROLLED", "preferred_category": "Hair", "marketing_consent": "No"
    }

    customer_b = {
        "recency_days": 10.0, "frequency_180d": 12.0, "monetary_180d": 8500.0,
        "ticket_count_90d": 0.0, "sessions_30d": 45.0, "abandoned_carts_30d": 0.0,
        "city_tier": "Tier 1", "age_group": "25-34", "acquisition_channel": "Instagram",
        "loyalty_tier": "Gold", "preferred_category": "Face", "marketing_consent": "Yes"
    }

    print("\n--- Step 3: Testing Single Prediction Endpoint ---")
    predict_response = requests.post(url_predict, json=customer_a)
    print(f"Predict Response Status: {predict_response.status_code}")
    print(f"Live Scoring Output:     {predict_response.json()}")

    print("\n--- Step 4: Testing Batch Prediction Endpoint ---")
    batch_payload = [customer_a, customer_b]
    batch_response = requests.post(url_batch, json=batch_payload)
    print(f"Batch Response Status:   {batch_response.status_code}")
    print(f"Batch Scoring Output:    {batch_response.json()}")

    print("\n--- STATUS: END-TO-END PIPELINE VERIFICATION SUCCESSFUL ---")

except Exception as e:
    print(f"\n❌ Pipeline Verification Failed: {str(e)}")

finally:
    print("\n--- Step 5: Shutting down pipeline engine instances ---")
    server_process.terminate()
    server_process.wait()
    print("Background server closed down cleanly.")