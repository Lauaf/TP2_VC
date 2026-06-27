# TP2 — Dog Breed Recognition

**IA 5.2 Computer Vision · 1° Cuatrimestre 2026**

Sistema de reconocimiento de razas de perros en tres etapas:

1. **Búsqueda por similitud** a partir de embeddings vectoriales.
2. **Clasificación supervisada** de la raza con modelos entrenados.
3. **Detección + clasificación** usando YOLOv8 y el clasificador de la Etapa 2.

---

## Resultados obtenidos

### Clasificación supervisada (test)

| Modelo | Accuracy | Precision | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 fine-tuned | 0.9429 | 0.9486 | 0.9429 | 0.9992 | 0.9432 |
| CNN custom | 0.5043 | 0.5177 | 0.5043 | 0.9928 | 0.4800 |

### Similitud (valid, `limit_per_class=3`, `top_k=10`)

| Modelo | Accuracy | NDCG@10 |
| --- | ---: | ---: |
| Baseline | 0.9429 | 0.9460 |
| ResNet18 fine-tuned | 0.9476 | 0.9429 |
| CNN custom | 0.5190 | 0.6572 |

---

## Requisitos

- Python 3.12
- Docker Desktop
- Dataset de Kaggle: [`gpiosenka/70-dog-breedsimage-data-set`](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)

---

## Inicio rápido con Docker

El modo por defecto usa `USE_PGVECTOR=false` y un índice JSON de demo (`data/embeddings.json`) para que la app levante sin reindexar PostgreSQL.

```bash
docker compose build
docker compose up
```

| Servicio | URL |
| --- | --- |
| Frontend | http://localhost:8080 |
| Backend | http://localhost:8000 |
| PostgreSQL | localhost:5432 |

Si algún puerto está ocupado:

```bash
# Linux/Mac
BACKEND_PORT=18000 FRONTEND_PORT=18080 POSTGRES_PORT_HOST=15432 docker compose up

# PowerShell
$env:BACKEND_PORT="18000"
$env:FRONTEND_PORT="18080"
$env:POSTGRES_PORT_HOST="15432"
docker compose up
```

---

## Desarrollo local

### 1. Descargar el dataset

```python
import kagglehub
path = kagglehub.dataset_download("gpiosenka/70-dog-breedsimage-data-set")
print(path)
```

Copiar `train/`, `valid/` y `test/` dentro de `data/dataset/`.

### 2. Crear entorno virtual e instalar dependencias

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```powershell
Copy-Item .env.local.example src\.env
```

### 4. Levantar backend y frontend

```powershell
# Backend
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir src --reload --port 8000

# Frontend (en otra terminal)
.venv\Scripts\python.exe -m uvicorn frontend.app:app --app-dir src --port 8080
```

---

## Scripts

Todos se ejecutan desde la raíz del repo.

### Construir índice de embeddings

```powershell
$env:EMBEDDING_MODEL="baseline"          # o resnet18_finetuned / cnn_custom
python scripts/build_index.py --split train
```

> Usar `USE_PGVECTOR=true` para indexar en PostgreSQL en vez del JSON.

### Entrenar clasificadores

```powershell
python scripts/train_classifier.py --model resnet18_finetuned
python scripts/train_classifier.py --model cnn_custom
```

Los checkpoints se guardan en `models/` (no versionado).

### Evaluar similitud

```powershell
python scripts/evaluate_similarity.py --model baseline --split valid --limit-per-class 3 --top-k 10
python scripts/evaluate_similarity.py --model resnet18_finetuned --split valid --limit-per-class 3 --top-k 10
python scripts/evaluate_similarity.py --model cnn_custom --split valid --limit-per-class 3 --top-k 10
```

### Generar assets del informe

```powershell
python scripts/generate_report_assets.py
```

Genera en `report_assets/`:

- `class_distribution.png`
- `classifier_history.png`
- `classifier_metrics.png`
- `similarity_metrics.png`
- `resnet_confusion_matrix.png`
- `cnn_confusion_matrix.png`
- `summary.json`

---

## Estructura del proyecto

```text
TP2_VC/
├── src/
│   ├── app/           # Backend FastAPI
│   ├── frontend/      # Frontend Gradio
│   └── lib/           # Servicios, modelos y utilidades
├── scripts/           # Scripts de entrenamiento, indexado y evaluación
├── data/
│   ├── dataset/       # Dataset (no versionado)
│   ├── external/      # Imágenes externas para evaluación
│   └── embeddings.json  # Índice demo chico
├── models/            # Checkpoints entrenados (no versionados)
├── output/            # Resultados y artefactos runtime (no versionados)
├── report_assets/     # Plots y resumen del informe (versionado)
├── etapa2_colab.ipynb
├── informe.ipynb
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.frontend
├── requirements.txt
└── requirements.backend.txt
```

---

## API del backend

| Método | Endpoint | Descripción |
| --- | --- | --- |
| `POST` | `/search` | Etapa 1: búsqueda por similitud |
| `POST` | `/classify` | Etapa 2: clasificación supervisada |
| `POST` | `/detect` | Etapa 3: detección + clasificación |
| `GET` | `/status/{job_id}` | Consultar estado de job asincrónico |

---

## Resultado esperado al probar

Con `data/dataset/valid/Beagle/01.jpg`:

- **Etapa 1**: `predicted_breed = Beagle`, vecinos con URLs `/files/data/...`
- **Etapa 2**: `breed = Beagle` con score alto usando `resnet18_finetuned`
- **Etapa 3**: al menos una detección y `detected_breeds = ["Beagle"]`

---

## Notebooks

| Notebook | Descripción |
| --- | --- |
| `etapa2_colab.ipynb` | Entrenamiento y evaluación pensado para Colab/GPU |
| `informe.ipynb` | Informe técnico completo: métricas, comparaciones y conclusiones |
