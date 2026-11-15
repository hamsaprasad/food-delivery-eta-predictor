import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import predict_eta, get_prediction_history, DeliveryFeatures
from src.database import init_db, get_connection

def main():
    print("=" * 65)
    print("RUNNING FASTAPI & SQLITE ENDPOINT VERIFICATION TESTS")
    print("=" * 65)

    # 1. Initialize SQLite Database
    init_db()
    print("[1] Database initialized.")

    # 2. Test POST /predict with default feature values (Normal ETA expected)
    print("\n[2] Testing POST /predict with default input features...")
    default_payload = DeliveryFeatures()
    res_normal = predict_eta(default_payload)
    print(f"  -> Predicted ETA : {res_normal.estimated_minutes:.2f} mins")
    print(f"  -> Delay Risk    : {res_normal.delay_risk}")
    print(f"  -> DB Record ID  : {res_normal.record_id}")
    print(f"  -> Status Message: {res_normal.message}")

    assert res_normal.estimated_minutes > 0, "ETA must be greater than 0"
    assert res_normal.delay_risk in ["Normal", "High"], "Delay risk must be Normal or High"
    assert res_normal.record_id is not None, "Record ID must be returned"
    print("  ✓ Test 1 passed: Normal prediction generated & stored.")

    # 3. Test POST /predict with high distance/heavy conditions (High Delay Risk expected)
    print("\n[3] Testing POST /predict with high distance & traffic conditions...")
    high_risk_payload = DeliveryFeatures(
        Road_Distance_km=60.0,
        Traffic_Level="Jam",
        Weather="Stormy",
        Vehicle_Type="Bicycle",
        Preparation_Time_Min=40,
        Number_of_Signals=18
    )
    res_high = predict_eta(high_risk_payload)
    print(f"  -> Predicted ETA : {res_high.estimated_minutes:.2f} mins")
    print(f"  -> Delay Risk    : {res_high.delay_risk}")
    print(f"  -> DB Record ID  : {res_high.record_id}")

    assert res_high.estimated_minutes > 90.0, "Expected ETA > 90 mins for high risk input"
    assert res_high.delay_risk == "High", "Expected delay_risk to be 'High'"
    print("  ✓ Test 2 passed: High delay risk (>90 min) correctly classified & stored.")

    # 4. Test GET /history (Retrieve 10 most recent predictions from SQLite)
    print("\n[4] Testing GET /history endpoint...")
    history = get_prediction_history(limit=10)
    print(f"  -> Retrieved {history['count']} recent predictions from SQLite:")
    for record in history["predictions"]:
        print(f"     ID #{record['id']} | {record['timestamp']} | "
              f"Dist: {record['road_distance_km']}km | "
              f"Prep: {record['preparation_time_min']}m | "
              f"ETA: {record['predicted_eta_mins']}m | "
              f"Risk: {record['delay_risk']}")

    assert history["count"] >= 2, "History must contain at least 2 records"
    assert history["predictions"][0]["id"] > history["predictions"][1]["id"], "History must be ordered by latest first"
    print("  ✓ Test 3 passed: History retrieved successfully in descending order.")

    # 5. Direct verification of the SQLite file
    print("\n[5] Verifying SQLite file on disk...")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        print(f"  -> Total rows in 'predictions' table: {count}")
    print("  ✓ Test 4 passed: Direct SQLite query verified.")

    print("\n" + "=" * 65)
    print("ALL FASTAPI & SQLITE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
