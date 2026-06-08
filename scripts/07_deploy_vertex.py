import os
import sys

from google.cloud import aiplatform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ID, REGION, MODELS_BUCKET

MODEL_ARTIFACT_URI = f"gs://{MODELS_BUCKET}/colbert/final/"


def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name="esci-search-colbert-ltr",
        artifact_uri=MODEL_ARTIFACT_URI,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-11:latest",
    )
    print(f"Model uploaded: {model.resource_name}")

    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
    )
    print(f"Endpoint: {endpoint.resource_name}")


if __name__ == "__main__":
    main()
