from google.cloud import storage


def download(gcs_uri: str, local_path: str) -> None:
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} -> {local_path}")


def upload(local_path: str, gcs_uri: str) -> None:
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> {gcs_uri}")
