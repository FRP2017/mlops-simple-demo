import os
import json
from kfp import dsl
from kfp.compiler import Compiler
from google.cloud import aiplatform

# --- Configuración del Pipeline ---
PROJECT_ID = os.environ.get("PROJECT_ID", "tu-gcp-project-id")
REGION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "tu-gcp-project-id-mlops-iris-bucket")
BUCKET_URI = f"gs://{BUCKET_NAME}"

PIPELINE_DOCKER_IMAGE = os.environ.get(
    "PIPELINE_DOCKER_IMAGE",
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-iris-models-repo/iris-classifier-app:latest"
)

PIPELINE_ROOT = f"{BUCKET_URI}/pipelines/runs"
PIPELINE_NAME = "iris-training-pipeline"
COMPILED_PIPELINE_PATH_LOCAL = "compiled_iris_pipeline.json"

# --- Definición de Componentes ---
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
    # Este es un placeholder para la ejecución del comando en KFP v1 style components
    # Para componentes ligeros de Python en KFP v2, la estructura sería diferente.
    # Aquí usamos una forma que es común para ejecutar scripts en contenedores.
    # El cuerpo real del componente se define por el comando y la imagen.
    # El código Python aquí es principalmente para definir la interfaz del componente.
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


# --- Definición del Pipeline ---
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Un pipeline de ejemplo para entrenar un modelo Iris.",
    pipeline_root=PIPELINE_ROOT
)
def iris_pipeline(
    raw_data_gcs_uri: str = f"{BUCKET_URI}/data/iris_raw.csv",
    test_split_ratio_param: float = 0.2,
    random_seed_param: int = 42,
    max_iter_param: int = 200
):
    preprocess_task = preprocess_data_op(
        raw_data_gcs_path=raw_data_gcs_uri,
        test_split_ratio=test_split_ratio_param,
        random_seed=random_seed_param
    )
    # El output de preprocess_task (output_processed_data_gcs_path)
    # se pasa implícitamente al input del siguiente componente si los nombres coinciden
    # o explícitamente como se muestra aquí.

    train_task = train_model_op(
        processed_data_gcs_path=preprocess_task.outputs["output_processed_data_gcs_path"],
        random_seed=random_seed_param,
        max_iter=max_iter_param
    )
    # Opcional: Añadir un paso para registrar el modelo en Vertex AI Model Registry.
    # from google_cloud_aiplatform.v1 import ModelUploadOp
    # model_upload = ModelUploadOp(
    #     project=PROJECT_ID,
    #     display_name="iris-classifier-from-pipeline",
    #     artifact_uri=train_task.outputs["output_model_gcs_path"], # Directorio del modelo
    #     serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest", # Ajustar framework
    #     # serving_container_predict_route="/predict", # Ajustar si es necesario
    #     # serving_container_health_route="/health",   # Ajustar si es necesario
    # ).after(train_task)


# --- Compilación (Opcional aquí, puede hacerse desde CLI o GitHub Actions) ---
if __name__ == '__main__':
    print(f"Compilando el pipeline: {PIPELINE_NAME}")
    print(f"  PROJECT_ID: {PROJECT_ID}")
    print(f"  REGION: {REGION}")
    print(f"  BUCKET_URI: {BUCKET_URI}")
    print(f"  PIPELINE_ROOT: {PIPELINE_ROOT}")
    print(f"  PIPELINE_DOCKER_IMAGE: {PIPELINE_DOCKER_IMAGE}")
    print(f"  Salida local: {COMPILED_PIPELINE_PATH_LOCAL}")

    # KFP v2 usa kfp.compiler.Compiler, pero para Vertex AI directamente con aiplatform SDK
    # la compilación está implícita al crear un PipelineJob o se puede usar el compilador de KFP.
    # Para generar un JSON que `gcloud ai pipeline-jobs submit` puede usar:
    Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=COMPILED_PIPELINE_PATH_LOCAL
    )
    print(f"Pipeline compilado en: {COMPILED_PIPELINE_PATH_LOCAL}")

    # Opcional: Subir y ejecutar directamente (requiere autenticación y configuración de la SDK)
    # print("Inicializando API de Vertex AI...")
    # aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
    #
    # print(f"Creando y ejecutando el PipelineJob para {PIPELINE_NAME}...")
    # job = aiplatform.PipelineJob(
    #     display_name=PIPELINE_NAME,
    #     template_path=COMPILED_PIPELINE_PATH_LOCAL,
    #     pipeline_root=PIPELINE_ROOT,
    #     # parameter_values={ # Si necesitas pasar parámetros
    #     #    'raw_data_gcs_uri': f"{BUCKET_URI}/data/iris_raw.csv",
    #     # }
    # )
    # job.run(sync=False) # sync=False para no esperar, True para esperar
    # print(f"PipelineJob enviado. Puedes verlo en la consola de Vertex AI. Job ID: {job.resource_name}")