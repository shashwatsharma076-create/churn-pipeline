import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    'raw_data'      : os.path.join(PROJECT_ROOT, 'data', 'raw', 'customer_churn.csv'),
    'clean_data'    : os.path.join(PROJECT_ROOT, 'data', 'processed', 'churn_clean.csv'),
    'simulated_data': os.path.join(PROJECT_ROOT, 'data', 'simulated'),
}

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}\n")
    for name, path in PATHS.items():
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{status} {name}")
        print(f"   {path}")