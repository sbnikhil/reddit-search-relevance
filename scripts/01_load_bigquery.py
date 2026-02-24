"""Step 1: Load ESCI data from GCS into BigQuery."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bigquery.load_esci import main

if __name__ == "__main__":
    main()
