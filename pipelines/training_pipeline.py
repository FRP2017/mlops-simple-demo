import os
import json
import time # Para el display_name si no vienen de GHA
from kfp import dsl
from kfp.compiler import Compiler
from google.cloud import aiplatform

# --- Configuración del Pipeline ---
# Usar os.getenv para obtener variables de entorno pasadas desde GitHub Actions
# Asegúrate que los nombres de las variables de entorno coincidan con los que exportas en GitHub Actions
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mlops-second")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
GCS_BUCKET_NAME_NO_GS = os.getenv("GCS_BUCKET_NAME_NO_GS", "mlops-second-iris-bucket")
BUCKET_URI = f"gs://{GCS_BUCKET_NAME_NO_GS}"

# Esta variable será seteada desde GitHub Actions con la imagen recién construida
PIPELINE_DOCKER_IMAGE = os.getenv(
    "PIPELINE_DOCKER_IMAGE", # Espera esta variable desde GitHub Actions
    f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/ml-models-repo/iris-classifier-app:latest" # Fallback
)

PIPELINE_ROOT = f"{BUCKET_URI}/pipelines/runs"
PIPELINE_NAME = "iris-training-pipeline"
COMPILED_PIPELINE_PATH_LOCAL = "compiled_iris_pipeline.json"

# --- Definición de Componentes (sin cambios respecto a tu original) ---
@dsl.component(
    base_image=PIPELINE_DOCKER_IMAGE
)
def preprocess_data_op(
    raw_data_gcs_path: str,
    output_processed_data_gcs_path: dsl.Output[dsl.Artifact],
    test_split_ratio: float = 0.2,
    random_seed: int = 42
):
    command = [
        "python", "/app/src/preprocess.py",
        "--input-data-path", raw_data_gcs_path,
        "--output-data-path", output_processed_data_gcs_path.path,
        "--test-split-ratio", str(test_split_ratio),
        "--random-seed", str(random_seed)
    ]
    print(f"Comando de preprocesamiento: {' '.join(command)}")


@dsl.component(
    base_image=PIPELINE_DOCKER_IMAGE
)
def train_model_op(
    processed_data_gcs_path: dsl.Input[dsl.Artifact],
    output_model_gcs_path: dsl.Output[dsl.Artifact],
    random_seed: int = 42,
    max_iter: int = 200
):
    command = [
        "python", "/app/src/train.py",
        "--input-data-path", processed_data_gcs_path.path,
        "--output-model-path", output_model_gcs_path.path,
        "--random-seed", str(random_seed),
        "--max-iter", str(max_iter)
    ]
    print(f"Comando de entrenamiento: {' '.join(command)}")


# --- Definición del Pipeline (sin cambios respecto a tu original) ---
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Un pipeline de ejemplo para entrenar un modelo Iris.",
    pipeline_root=PIPELINE_ROOT
)
def iris_pipeline(
    raw_data_gcs_uri: str = f"{BUCKET_URI}/data/iris_raw.csv", # Parámetro con default
    test_split_ratio_param: float = 0.2,
    random_seed_param: int = 42,
    max_iter_param: int = 200
):
    preprocess_task = preprocess_data_op(
        raw_data_gcs_path=raw_data_gcs_uri,
        test_split_ratio=test_split_ratio_param,
        random_seed=random_seed_param
    )
    train_task = train_model_op(
        processed_data_gcs_path=preprocess_task.outputs["output_processed_data_gcs_path"],
        random_seed=random_seed_param,
        max_iter=max_iter_param
    )

# --- Compilación y Envío del Pipeline ---
if __name__ == '__main__':
    print("--- Iniciando Compilación y Envío del Pipeline ---")
    print(f"Usando Configuración:")
    print(f"  GCP_PROJECT_ID: {GCP_PROJECT_ID}")
    print(f"  GCP_REGION: {GCP_REGION}")
    print(f"  GCS_BUCKET_NAME_NO_GS: {GCS_BUCKET_NAME_NO_GS}")
    print(f"  BUCKET_URI: {BUCKET_URI}")
    print(f"  PIPELINE_ROOT: {PIPELINE_ROOT}")
    print(f"  PIPELINE_DOCKER_IMAGE: {PIPELINE_DOCKER_IMAGE}")
    print(f"  Salida JSON local: {COMPILED_PIPELINE_PATH_LOCAL}")

    # 1. Compilación del Pipeline (como estaba en tu script)
    print(f"\nCompilando el pipeline '{PIPELINE_NAME}'...")
    Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=COMPILED_PIPELINE_PATH_LOCAL
    )
    print(f"Pipeline compilado exitosamente en: {COMPILED_PIPELINE_PATH_LOCAL}")

    # 2. Envío del Pipeline a Vertex AI (NUEVA SECCIÓN)
    print("\nEnviando el pipeline a Vertex AI...")

    # Obtener Service Account para la ejecución del pipeline y GITHUB_RUN_ID/SHA para el display_name
    # Estos deben ser pasados como variables de entorno desde GitHub Actions
    pipeline_run_service_account = os.getenv("GCP_SA_EMAIL_FOR_PIPELINE")
    github_run_id = os.getenv("GITHUB_RUN_ID", f"local-{time.strftime('%Y%m%d-%H%M%S')}")
    github_sha_short = os.getenv("GITHUB_SHA", "manual")[:7] # Usar los primeros 7 chars del SHA

    display_name = f"iris-pipeline-gh-{github_run_id}-{github_sha_short}"

    # Parámetros para la ejecución del pipeline.
    # Estos deben coincidir con los parámetros de tu función `iris_pipeline`.
    # El valor de `raw_data_gcs_uri` es el único que tu `gcloud` estaba sobreescribiendo.
    # Puedes añadir más aquí si necesitas sobreescribir otros defaults desde GHA.
    pipeline_parameters = {
        "raw_data_gcs_uri": f"gs://{GCS_BUCKET_NAME_NO_GS}/data/iris_raw.csv",
        # "test_split_ratio_param": 0.25, # Ejemplo si quisieras sobreescribir
        # "random_seed_param": 123,      # Ejemplo
        # "max_iter_param": 300          # Ejemplo
    }
    print(f"  Display Name: {display_name}")
    print(f"  Pipeline Root: {PIPELINE_ROOT}")
    print(f"  Service Account (para ejecución del pipeline): {pipeline_run_service_account}")
    print(f"  Parámetros del Pipeline: {pipeline_parameters}")


    if not pipeline_run_service_account:
        print("Error: La variable de entorno 'GCP_SA_EMAIL_FOR_PIPELINE' no está configurada. Esta es necesaria para enviar el job.")
        exit(1)

    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, staging_bucket=PIPELINE_ROOT)

    # Crear el PipelineJob
    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=COMPILED_PIPELINE_PATH_LOCAL, # El JSON compilado localmente
        pipeline_root=PIPELINE_ROOT,
        parameter_values=pipeline_parameters,
        enable_caching=False # O True, según tu preferencia
    )

    # Enviar el PipelineJob
    print(f"Enviando PipelineJob a Vertex AI...")
    job.submit(
        service_account=pipeline_run_service_account # SA que usará Vertex AI para ejecutar el pipeline
    )

    print(f"\nPipeline Job '{job.resource_name}' enviado exitosamente a Vertex AI.")
    print(f"Puedes monitorear el job en la consola de GCP o aquí: {job._dashboard_uri()}")
    print("--- Fin del Script de Compilación y Envío ---")