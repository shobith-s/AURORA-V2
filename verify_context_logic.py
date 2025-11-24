import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_context_logic():
    print("Testing Context-Aware Preprocessing Logic...")
    
    # Test Case 1: Numeric Column (Revenue)
    numeric_data = [100, 150, 200, 250, 300, 1000, 5000] 
    col_name = "revenue"
    
    print(f"\n1. Testing Numeric Column: {col_name}")
    
    # General Context
    try:
        resp_general = requests.post(f"{BASE_URL}/preprocess", json={
            "column_data": numeric_data,
            "column_name": col_name,
            "context": "general"
        })
        print(f"   [GENERAL] Status: {resp_general.status_code}")
        if resp_general.status_code == 200:
            res_general = resp_general.json()
            print(f"   [GENERAL] Response: {json.dumps(res_general, indent=2)}")
        else:
            print(f"   [GENERAL] Error: {resp_general.text}")
    except Exception as e:
        print(f"   [GENERAL] Failed: {e}")

    # Regression Context
    try:
        resp_regression = requests.post(f"{BASE_URL}/preprocess", json={
            "column_data": numeric_data,
            "column_name": col_name,
            "context": "regression"
        })
        print(f"   [REGRESSION] Status: {resp_regression.status_code}")
        if resp_regression.status_code == 200:
            res_regression = resp_regression.json()
            print(f"   [REGRESSION] Response: {json.dumps(res_regression, indent=2)}")
            
            if res_regression.get('action') == 'standard_scale' and "context_bias" in res_regression.get('source', ''):
                print("   SUCCESS: Regression context correctly enforced scaling.")
        else:
            print(f"   [REGRESSION] Error: {resp_regression.text}")
            
    except Exception as e:
        print(f"   [REGRESSION] Failed: {e}")

    # Test Case 2: Target Column (Classification)
    target_data = ["A", "B", "A", "C", "B", "A"]
    target_col = "target"
    
    print(f"\n2. Testing Target Column: {target_col}")
    
    # General Context
    try:
        resp_general = requests.post(f"{BASE_URL}/preprocess", json={
            "column_data": target_data,
            "column_name": target_col,
            "context": "general"
        })
        print(f"   [GENERAL] Status: {resp_general.status_code}")
        if resp_general.status_code == 200:
            res_general = resp_general.json()
            print(f"   [GENERAL] Response: {json.dumps(res_general, indent=2)}")
        else:
            print(f"   [GENERAL] Error: {resp_general.text}")
    except Exception as e:
        print(f"   [GENERAL] Failed: {e}")

    # Classification Context
    try:
        resp_class = requests.post(f"{BASE_URL}/preprocess", json={
            "column_data": target_data,
            "column_name": target_col,
            "context": "classification"
        })
        print(f"   [CLASSIFICATION] Status: {resp_class.status_code}")
        if resp_class.status_code == 200:
            res_class = resp_class.json()
            print(f"   [CLASSIFICATION] Response: {json.dumps(res_class, indent=2)}")
            
            if res_class.get('action') == 'label_encode' and "context_bias" in res_class.get('source', ''):
                print("   SUCCESS: Classification context enforced Label Encoding on target.")
        else:
            print(f"   [CLASSIFICATION] Error: {resp_class.text}")

    except Exception as e:
        print(f"   [CLASSIFICATION] Failed: {e}")

if __name__ == "__main__":
    time.sleep(1)
    test_context_logic()
