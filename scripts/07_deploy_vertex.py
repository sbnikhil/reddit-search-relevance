from google.cloud import aiplatform

PROJECT_ID = "reddit-search-relevance-485717"
REGION = "us-central1"
MODEL_ARTIFACT_URI = "gs://reddit-search-relevance-models/colbert/final/"


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
