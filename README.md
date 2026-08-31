# MTU CEIT Student Academic Support System

A complete, local-first academic decision-support application for the Computer Engineering and Information Technology (CEIT) department at Mandalay Technological University. This first pilot is deliberately limited to authorized CEIT teachers and these seven academic levels: First Year, Second Year, Third Year, Fourth Year, Fifth Year First Semester, Fifth Year Second Semester, and Final Year.

It combines:

- Role-based administrator and teacher accounts with server-side sessions.
- Persistent student profiles identified by non-identifying institutional codes.
- Individual and batch prediction, risk classification, local feature-influence explanations, and prediction history.
- Six-year cohort analytics, early-intervention tracking, CSV reporting, audit logs, and model governance metrics.
- A reproducible UCI benchmark comparing ANN, Logistic Regression, and Random Forest, including gender and family-support bias screening.
- A secure Python/FastAPI/SQLite backend and responsive browser interface.

The ANN predicts either:

- **Pass / fail** (binary classification), with accuracy, precision, recall, F1-score, and a confusion matrix.
- **Final grade** (regression), with MAE, RMSE, R², and an easy-to-read within-margin accuracy.

The network follows the requested architecture: input layer → 64-neuron ReLU layer → 32-neuron ReLU layer → one output. Dropout, L2 regularization, validation-based early stopping, and learning-rate reduction help limit overfitting.

## Setup

Python 3.10–3.12 is required because TensorFlow support commonly lags the newest Python release. From this directory:

### Team quick start (fresh clone)

Install Python 3.10–3.12 and the current Node.js LTS release, then run:

```bash
./scripts/team_first_run.sh
./scripts/start_local_web.sh
```

The first command creates the local environments, generates the clearly labelled synthetic CEIT dataset, trains the demonstration ANN, prompts you to choose a local administrator password, and seeds demonstration records. Open `http://localhost:3000` and sign in with `admin@mtu.local` and the password you chose. Generated data, model artifacts, the database, and passwords are not committed to Git.

### Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

TensorFlow availability depends on the Python version and platform. On a supported Apple Silicon environment, `tensorflow` uses the native macOS package supplied by current TensorFlow releases.

## Dataset

Place a CSV in `data/`. The loader accepts comma- or semicolon-separated files. Canonical columns are:

| Feature | Accepted examples | Valid range |
|---|---|---|
| Gender | `gender`, UCI `sex` | category |
| Age | `age` | 10–100 |
| Attendance | `attendance`, `attendance_percent` | 0–100 |
| Study time | `study_time`, UCI `studytime` | 0–168 |
| Previous grade | `previous_grade`, UCI `G2` | 0–100 |
| Internet access | `internet_access`, UCI `internet` | yes/no |
| Family support | `family_support`, UCI `famsup` | yes/no |
| Absences | `absences` | 0–365 |
| Participation | `participation`, UCI `activities` | 0–100 or yes/no for activities |
| Homework completion | `homework_completion` | 0–100 |
| Target | `final_grade`, UCI `G3`, `performance_index` | numeric |

The original UCI dataset does not contain every requested field. Attendance is transparently estimated from absences and a 200-day school year; unavailable fields are reported and excluded. For best results, use a dataset containing the real requested measurements rather than inferred values.

To test the pipeline without personal data, create synthetic demo data:

```bash
python scripts/generate_demo_data.py --rows 1000
```

Synthetic data only proves that the software runs; it does **not** prove real-world predictive quality.

### CEIT project dataset (synthetic)

Because approved real MTU records are not available for this project, the operational demonstration model uses a clearly labelled synthetic CEIT dataset. It contains 1,400 records—200 for each of the seven configured academic levels—with no names or real student identifiers.

Generate the exact dataset again with the fixed project seed:

```bash
python scripts/generate_ceit_dataset.py \
  --records-per-level 200 \
  --seed 20260831 \
  --output data/ceit_synthetic_students.csv
```

The dataset has 968 pass and 432 fail records using the project threshold of 50. Its generated relationships include random noise and gender has no direct effect in the outcome equation. The formatted research workbook with data dictionary, assumptions, formula-driven summary, and chart is saved at `outputs/ceit-synthetic-dataset/CEIT_Synthetic_Student_Performance_Dataset.xlsx`. Full ANN/baseline tables, graphs, bias screening, and discussion are in [reports/CEIT_SYNTHETIC_EXPERIMENT.md](reports/CEIT_SYNTHETIC_EXPERIMENT.md).

Train the operational demonstration ANN:

```bash
python -m student_performance.train \
  --data data/ceit_synthetic_students.csv \
  --dataset-label "Synthetic MTU CEIT Project Dataset" \
  --synthetic-data \
  --task classification \
  --pass-threshold 50 \
  --output artifacts/classifier
```

Current held-out ANN results are accuracy 0.761, precision 0.890, recall 0.747, and F1 0.812. The majority-class baseline accuracy is 0.693. These figures must be described as a **synthetic CEIT project experiment**, never as evidence about real MTU students.

### Official UCI benchmark

Download the official UCI Student Performance files and run the reproducible experiment:

```bash
python scripts/download_uci.py
python -m student_performance.experiment \
  --data data/uci/student-mat.csv \
  --dataset-name "UCI Student Performance — Mathematics" \
  --pass-threshold 10 \
  --output artifacts/experiments/uci
```

The current fixed held-out split contains 252 training, 64 validation, and 79 test records. Results are:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| ANN (64–32) | 0.835 | 0.935 | 0.811 | 0.869 | 0.941 |
| Logistic Regression | 0.848 | 0.956 | 0.811 | 0.878 | **0.955** |
| Random Forest | **0.873** | **0.957** | **0.849** | **0.900** | 0.916 |

These are UCI Mathematics results, not validated MTU performance. The complete result tables, graphs, bias audit, and discussion are in [reports/CEIT_UCI_EXPERIMENT.md](reports/CEIT_UCI_EXPERIMENT.md). The source is the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance), DOI [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).

### Approved MTU CEIT validation (pending)

MTU validation must wait for a formally approved, anonymized CEIT export. Store it in an access-controlled location and run it into a separate output directory, for example:

```bash
python -m student_performance.experiment \
  --data data/approved_mtu_ceit.csv \
  --dataset-name "Approved MTU CEIT Dataset" \
  --pass-threshold 50 \
  --output artifacts/experiments/mtu
```

Adapt the pass threshold to the approved MTU policy. Do not include names, phone numbers, addresses, or unrelated personal data. Approval, field definitions, consent/legal basis, retention period, and authorized recipients should be recorded before import.

## Train and evaluate

Classification (a percentage-grade dataset defaults to a passing grade of 50):

```bash
python -m student_performance.train \
  --data data/demo_students.csv \
  --task classification \
  --pass-threshold 50 \
  --output artifacts/classifier
```

Regression:

```bash
python -m student_performance.train \
  --data data/demo_students.csv \
  --task regression \
  --output artifacts/regressor
```

For UCI's 0–20 `G3` scale, the automatic classification threshold is 10. Override it with `--pass-threshold` if the institution uses another policy.

Training writes a Keras model, JSON preprocessing state, metadata, evaluation metrics, plots, and a SHA-256 integrity manifest under the selected artifact directory. Preprocessing learns only from the training split, preventing test-data leakage.

## Predict

Edit `examples/student.json`, keeping values within the documented ranges, then run:

```bash
python -m student_performance.predict \
  --artifacts artifacts/classifier \
  --input examples/student.json
```

The JSON file can contain one student object or a list of up to 10,000 objects.

## Local web application

The browser interface is in `web/`. It runs locally and connects only to the loopback-only Python API by default, so student inputs are not sent to an external server.

Install the frontend once:

```bash
cd web
npm install
cd ..
```

Initialize the first administrator. The command prompts securely for a password of at least 12 characters:

```bash
python -m student_performance.manage create-user \
  --email admin@mtu.local \
  --name "MTU Administrator" \
  --role admin
```

Optional: populate the empty system with clearly labelled, synthetic demonstration records:

```bash
python scripts/seed_demo_system.py --rows 18
```

After training `artifacts/classifier`, start both the API and website:

```bash
./scripts/start_local_web.sh
```

Open `http://localhost:3000`. Stop both services with `Control-C`. The API binds to `127.0.0.1`, applies strict schema validation, limits request size and rate, disables response caching, and accepts browser calls only from the two documented local origins.

The application provides Overview, Students, Prediction, Interventions, Model & Reports, User Management, and Security Settings screens. Administrators can create teacher accounts. Each user should change a temporary password immediately from Security Settings.

For batch import, use the column order shown in `examples/students_batch.csv`. The major is locked to CEIT; it is not a CSV column. The browser validates the seven supported year/semester combinations, accepts up to 500 records per import, and skips duplicate student codes.

For separate terminals, run these from the activated Python environment:

```bash
python -m student_performance.api
```

```bash
cd web
npm run dev
```

## Notebook

Launch `jupyter lab` and open `notebooks/student_performance_ann.ipynb`. The notebook uses the same tested package rather than duplicating training logic.

## Security and responsible use

- CSV and prediction files have size/row limits; numeric fields have strict ranges.
- Predictions accept JSON data only. Model preprocessing is stored as readable JSON—no untrusted pickle deserialization.
- Keras artifacts are hash-checked and loaded in safe mode. Keep the whole artifact directory access-controlled; the manifest detects accidental changes, not a malicious person who can replace both files and hashes.
- The system stores pseudonymous student codes but not student names. Passwords use salted PBKDF2-HMAC-SHA256 hashes; session tokens are stored server-side as hashes and expire after eight hours.
- SQLite uses foreign keys, integrity constraints, WAL mode, transaction rollback, query indexes, and an audit trail. Keep `data/student_support.db` readable only by authorized operating-system accounts and include it in an approved encrypted backup process.
- Protect educational records with least-privilege access, encryption in transit/at rest, retention limits, audit logs, and applicable local privacy rules.
- Gender and family support can create unfair outcomes. Measure performance by relevant demographic groups, document limitations, and never use a prediction as the sole basis for punishment, grading, or denying opportunity. Use it to offer supportive human review.
- “High accuracy” is not guaranteed. Report held-out metrics, compare against a simple baseline, monitor drift, and retrain only with approved, representative data.

## Tests

```bash
pytest
```

Frontend validation:

```bash
cd web
npm run lint
npm run build
npm audit
```
