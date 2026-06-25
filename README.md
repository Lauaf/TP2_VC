# TP2 - Dog Breed Recognition

Trabajo practico de Vision por Computadora basado en tres etapas:

1. Busqueda por similitud a partir de embeddings.
2. Clasificacion supervisada de la raza.
3. Deteccion de perros + clasificacion por recorte.

El repo queda listo para correr en local o con Docker, con notebooks y assets del informe ya preparados.

## Estado del TP

- Etapa 1 implementada y validada end-to-end.
- Etapa 2 implementada, entrenada y evaluada con `resnet18_finetuned` y `cnn_custom`.
- Etapa 3 implementada y validada end-to-end.
- Frontend Gradio funcionando como cliente HTTP del backend.
- Docker validado con `docker compose build` y `docker compose up`.
- Notebook de Colab e informe actualizados.

## Estructura

```text
TP2_VC/
|-- src/
|   |-- app/
|   |-- frontend/
|   `-- lib/
|-- scripts/
|-- data/
|   |-- dataset/                # no versionado
|   `-- embeddings.json         # indice JSON chico para demo
|-- models/                     # checkpoints locales (no versionados)
|-- output/                     # resultados y artefactos runtime (no versionados)
|-- report_assets/              # plots y resumen del informe
|-- etapa2_colab.ipynb
|-- informe.ipynb
|-- docker-compose.yml
|-- Dockerfile
|-- Dockerfile.frontend
|-- requirements.txt
`-- requirements.backend.txt
```

## Requisitos

- Python 3.12
- Docker Desktop
- Dataset de Kaggle `gpiosenka/70-dog-breedsimage-data-set`

## Dataset

Descargar con:

```python
import kagglehub

path = kagglehub.dataset_download("gpiosenka/70-dog-breedsimage-data-set")
print(path)
```

Luego copiar `train/`, `valid/` y `test/` dentro de `data/dataset/`.

## Demo rapida con Docker

El modo por defecto usa `USE_PGVECTOR=false` y un indice JSON chico (`data/embeddings.json`) para que la app levante sin tener que reindexar PostgreSQL primero.

```bash
docker compose build
docker compose up
```

Servicios:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8080`
- PostgreSQL: `localhost:5432`

Si alguno de esos puertos esta ocupado, se pueden overridear:

```bash
BACKEND_PORT=18000 FRONTEND_PORT=18080 POSTGRES_PORT_HOST=15432 docker compose up
```

En PowerShell:

```powershell
$env:BACKEND_PORT="18000"
$env:FRONTEND_PORT="18080"
$env:POSTGRES_PORT_HOST="15432"
docker compose up
```

## Desarrollo local

### 1. Crear y activar venv

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Configurar entorno local

```powershell
Copy-Item .env.local.example src\.env
```

### 4. Levantar backend

```powershell
cd ..
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir src --reload --port 8000
```

### 5. Levantar frontend

```powershell
cd ..
.venv\Scripts\python.exe -m uvicorn frontend.app:app --app-dir src --port 8080
```

## Scripts utiles

Todos se corren desde la raiz del repo.

### Construir indice de embeddings

```powershell
python scripts/build_index.py --split train
```

Variables importantes:

- `EMBEDDING_MODEL=baseline`
- `EMBEDDING_MODEL=resnet18_finetuned`
- `EMBEDDING_MODEL=cnn_custom`
- `USE_PGVECTOR=false|true`

El script guarda rutas relativas al repo para que el indice sea portable entre maquinas.

### Entrenar clasificadores

```powershell
python scripts/train_classifier.py --model resnet18_finetuned
python scripts/train_classifier.py --model cnn_custom
```

### Evaluar similitud

```powershell
python scripts/evaluate_similarity.py --model baseline --split valid --limit-per-class 3 --top-k 5
python scripts/evaluate_similarity.py --model resnet18_finetuned --split valid --limit-per-class 3 --top-k 5
python scripts/evaluate_similarity.py --model cnn_custom --split valid --limit-per-class 3 --top-k 5
```

### Generar assets para el informe

```powershell
python scripts/generate_report_assets.py
```

Genera:

- `report_assets/classifier_history.png`
- `report_assets/classifier_metrics.png`
- `report_assets/similarity_metrics.png`
- `report_assets/resnet_confusion_matrix.png`
- `report_assets/cnn_confusion_matrix.png`
- `report_assets/summary.json`

## Modelos

Se soportan tres nombres de modelo:

- `baseline`
- `resnet18_finetuned`
- `cnn_custom`

El backend usa:

- `POST /search` para Etapa 1
- `POST /classify` para Etapa 2
- `POST /detect` para Etapa 3
- `GET /status/{job_id}` para consultar el trabajo asincronico

## Resultado esperado al probar

Con una imagen como `data/dataset/valid/Beagle/01.jpg` deberias ver:

- En Etapa 1: `predicted_breed = Beagle` y vecinos con URLs `/files/data/...`.
- En Etapa 2: `breed = Beagle` con score alto usando `resnet18_finetuned`.
- En Etapa 3: al menos una deteccion y `detected_breeds = ["Beagle"]`.

## Resultados finales obtenidos

### Clasificacion supervisada en test

- `resnet18_finetuned`
  - accuracy: `0.9429`
  - precision: `0.9486`
  - recall: `0.9429`
  - specificity: `0.9992`
  - f1: `0.9432`
- `cnn_custom`
  - accuracy: `0.2457`
  - precision: `0.2908`
  - recall: `0.2457`
  - specificity: `0.9891`
  - f1: `0.2217`

### Similitud en valid (`limit_per_class=3`, `top_k=5`)

- `baseline`: `0.9286`
- `resnet18_finetuned`: `0.9619`
- `cnn_custom`: `0.3238`

## Notebooks

- `etapa2_colab.ipynb`
  - flujo de entrenamiento y evaluacion pensado para Colab/GPU
- `informe.ipynb`
  - resumen tecnico del trabajo, metricas, comparaciones y conclusiones

## Notas

- `output/` y `models/` no se versionan.
- `report_assets/` si se versiona porque forma parte del material del informe.
- `data/embeddings.json` es un indice de demo chico. Para evaluaciones serias conviene reindexar completo.
- Si quieren usar PostgreSQL + pgvector para la Etapa 1, cambiar `USE_PGVECTOR=true` y volver a correr `scripts/build_index.py`.
