
from pathlib import Path
import re
import uuid
import hashlib
import secrets
import sqlite3
import time
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from scipy import ndimage as ndi

from src.vessels import segment_vessels, skeleton_of
from src.features import (baseline_features, topology_features, BASE_NAMES,
                          TOPO_NAMES)
from src.preprocessing import load_octa_image
from src.classification import apply_fov_calibrators
from src.cnn_embeddings import FrozenResNet18, EMBEDDING_DIM
from src.hybrid import CNN_NAMES, MODEL_VARIANTS, transform_variant

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "web_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)
AUTH_DB = ROOT / "outputs" / "accounts.sqlite3"
TOKEN_TTL = 60 * 60 * 24 * 14


def db_connect():
    connection = sqlite3.connect(AUTH_DB)
    connection.row_factory = sqlite3.Row
    return connection


with db_connect() as connection:
    connection.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, salt TEXT NOT NULL,
        created_at INTEGER NOT NULL)""")
    connection.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
        expires_at INTEGER NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))""")


class AuthInput(BaseModel):
    name: str = Field(default="", max_length=80)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


def safe_user(user):
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def create_session(user_id):
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode()).hexdigest()
    with db_connect() as connection:
        connection.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                           (digest, user_id, int(time.time()) + TOKEN_TTL))
    return token


def current_user(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Please sign in to continue")
    digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
    now = int(time.time())
    with db_connect() as connection:
        row = connection.execute("""SELECT users.* FROM sessions
            JOIN users ON users.id=sessions.user_id
            WHERE sessions.token_hash=? AND sessions.expires_at>?""",
            (digest, now)).fetchone()
    if row is None:
        raise HTTPException(401, "Your session expired. Please sign in again")
    return row

app = FastAPI(title="OCTA topology research API")


@app.exception_handler(RequestValidationError)
async def validation_error_message(request: Request, error: RequestValidationError):
    issues = []
    for item in error.errors():
        field = ".".join(str(part) for part in item.get("loc", [])[1:]) or "Input"
        issues.append(f"{field}: {item.get('msg', 'invalid value')}")
    return JSONResponse(status_code=422, content={"detail": ". ".join(issues)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=str(OUTPUT)), name="results")
ARTIFACTS = ROOT / "outputs"


_classifier_cache = {"mtime_ns": None, "model": None}
_cnn_encoder_cache = {"mtime_ns": None, "encoder": None}


def load_classifier():
    model_path = ARTIFACTS / "dr_classifier.pkl"
    if not model_path.is_file():
        return None
    modified = model_path.stat().st_mtime_ns
    if _classifier_cache["mtime_ns"] != modified:
        with model_path.open("rb") as handle:
            _classifier_cache["model"] = pickle.load(handle)
        _classifier_cache["mtime_ns"] = modified
    return _classifier_cache["model"]


def load_cnn_encoder():
    classifier = load_classifier()
    if classifier is None:
        return None
    modified = _classifier_cache["mtime_ns"]
    if _cnn_encoder_cache["mtime_ns"] != modified:
        _cnn_encoder_cache["encoder"] = FrozenResNet18(
            state_dict=classifier["cnn_state_dict"]
        )
        _cnn_encoder_cache["mtime_ns"] = modified
    return _cnn_encoder_cache["encoder"]

@app.get("/artifacts/{filename}")
def artifact_file(filename: str):
    fixed_files = {
        "topology_features.csv", "topology_features_500.csv", "vessel_density.png", "topology_analysis.png",
        "pipeline_examples.png", "cm_baseline.png",
        "cm_baseline_topology.png", "feature_importance.png",
        "cm_binary_baseline.png", "cm_binary_baseline_topology.png",
        "cm_multiclass_baseline.png", "cm_multiclass_baseline_topology.png",
        "cm_binary_cnn_baseline.png", "cm_binary_topology_only.png", "cm_binary_hybrid.png",
        "cm_multiclass_cnn_baseline.png", "cm_multiclass_topology_only.png", "cm_multiclass_hybrid.png",
    }
    if filename not in fixed_files and not re.fullmatch(r"comparison_\d+\.png", filename):
        raise HTTPException(404, "Output file was not found")
    path = ARTIFACTS / filename
    if not path.is_file():
        raise HTTPException(404, "Output file was not found")
    return FileResponse(path)


@app.get("/")
def home():
    return {"message": "OCTA topology analysis API is running"}


@app.post("/api/auth/signup")
def signup(payload: AuthInput):
    name = payload.name.strip()
    email = payload.email.strip().lower()
    if not name:
        raise HTTPException(400, "Please enter your name")
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise HTTPException(400, "Please enter a valid email address")
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", payload.password.encode(), bytes.fromhex(salt), 310_000
    ).hex()
    try:
        with db_connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users(name,email,password_hash,salt,created_at) VALUES(?,?,?,?,?)",
                (name, email, password_hash, salt, int(time.time())),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists")
    with db_connect() as connection:
        user = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return {"token": create_session(user_id), "user": safe_user(user)}


@app.post("/api/auth/login")
def login(payload: AuthInput):
    email = payload.email.strip().lower()
    with db_connect() as connection:
        user = connection.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if user is None:
        raise HTTPException(401, "Email or password is incorrect")
    candidate = hashlib.pbkdf2_hmac(
        "sha256", payload.password.encode(), bytes.fromhex(user["salt"]), 310_000
    ).hex()
    if not secrets.compare_digest(candidate, user["password_hash"]):
        raise HTTPException(401, "Email or password is incorrect")
    return {"token": create_session(user["id"]), "user": safe_user(user)}


@app.get("/api/auth/me")
def me(authorization: str | None = Header(default=None)):
    return {"user": safe_user(current_user(authorization))}


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
        with db_connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash=?", (digest,))
    return {"success": True}


@app.get("/api/summary")
def summary(authorization: str | None = Header(default=None)):
    """Return metrics from the completed OCTA500 feature extraction run."""
    current_user(authorization)
    import json
    import pandas as pd

    results_path = ARTIFACTS / "results.json"
    features_path = ARTIFACTS / "topology_features_500.csv"
    if not features_path.exists():
        features_path = ARTIFACTS / "topology_features.csv"
    if not results_path.exists() or not features_path.exists():
        raise HTTPException(404, "Completed dataset outputs were not found")

    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    frame = pd.read_csv(features_path)
    topology_names = [
        "vessel_density",
        "skel_density", "endpoints_k", "branchpoints_k", "n_components",
        "largest_comp_frac", "mean_comp_size", "euler_number",
        "mean_segment_len", "max_avascular_gap", "frac_far_from_vessel",
    ]
    means = {name: float(frame[name].mean()) for name in topology_names if name in frame}
    validation = {}
    for vessel in ("artery", "vein", "capillary"):
        col = f"{vessel}_dice"
        if col in frame:
            validation[vessel] = float(frame[col].mean())

    feature_columns = [
        name for name in [
            "id", "vessel_density", "skeleton_pixels", *TOPO_NAMES,
            "artery_dice", "vein_dice", "capillary_dice",
        ] if name in frame.columns
    ]
    feature_rows = []
    for _, row in frame[feature_columns].iterrows():
        feature_rows.append({
            name: (str(value) if name == "id" else
                   None if pd.isna(value) else float(value))
            for name, value in row.items()
        })

    actual_artifacts = [
        path.name for path in sorted(ARTIFACTS.glob("comparison_*.png"))
    ]
    actual_artifacts += [
        name for name in (
            "vessel_density.png", "topology_analysis.png", "cm_baseline.png",
            "cm_baseline_topology.png", "feature_importance.png",
            "cm_binary_baseline.png", "cm_binary_baseline_topology.png",
            "cm_multiclass_baseline.png", "cm_multiclass_baseline_topology.png",
            "cm_binary_cnn_baseline.png", "cm_binary_topology_only.png", "cm_binary_hybrid.png",
            "cm_multiclass_cnn_baseline.png", "cm_multiclass_topology_only.png", "cm_multiclass_hybrid.png",
        )
        if (ARTIFACTS / name).is_file()
    ]
    demo_artifacts = [
        name for name in (
            "pipeline_examples.png",
        ) if (ARTIFACTS / name).is_file()
    ]

    return {
        **results,
        "topology_means": means,
        "validation_dice": validation,
        "samples": frame[["id", "vessel_density", "skeleton_pixels"]].head(10).to_dict(orient="records"),
        "feature_rows": feature_rows,
        "artifacts": {
            "octa500": actual_artifacts,
            "synthetic_demo": demo_artifacts,
        },
        "classification_available": load_classifier() is not None,
        "classification_metrics": results.get("tasks", {}),
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    fov: str = Form(default="6mm"),
    authorization: str | None = Header(default=None),
):
    current_user(authorization)
    if fov not in {"3mm", "6mm"}:
        raise HTTPException(422, "FOV must be either '3mm' or '6mm'")
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(400, "Unsupported image format")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Uploaded file is empty")

        image_id = uuid.uuid4().hex[:10]
        input_path = OUTPUT / f"{image_id}_original.png"

        with open(input_path, "wb") as f:
            f.write(contents)

        try:
            image = load_octa_image(input_path, size=256)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

        mask = segment_vessels(image)
        skeleton = skeleton_of(mask)

        features = topology_features(mask, skeleton)
        measurements = {
            name: float(value)
            for name, value in zip(TOPO_NAMES, features)
        }

        classifier = load_classifier()
        classification = None
        if classifier is not None:
            embedding = load_cnn_encoder().embed(image)[0]
            base = baseline_features(image, mask)
            fov_mm = float(fov.removesuffix("mm"))
            model_features = dict(zip(BASE_NAMES, base))
            model_features.update(measurements)
            model_features.update({name: float(value) for name, value in zip(CNN_NAMES, embedding)})
            model_features["fov_mm"] = fov_mm
            mm_per_pixel = fov_mm / image.shape[0]
            model_features["endpoints_k"] *= 1000 / (mm_per_pixel ** 2)
            model_features["branchpoints_k"] *= 1000 / (mm_per_pixel ** 2)
            model_features["mean_comp_size"] *= mm_per_pixel ** 2
            model_features["mean_segment_len"] *= mm_per_pixel
            model_features["max_avascular_gap"] *= mm_per_pixel
            model_features["frac_far_from_vessel"] = float(
                (ndi.distance_transform_edt(~mask) > 5 * 6 / image.shape[0]).mean()
            )
            model_features["fov"] = fov
            classification = {}
            class_labels = {"binary": ["NORMAL", "DR"],
                             "multiclass": ["NORMAL", "DR", "OTHER"]}
            for task, task_data in classifier["tasks"].items():
                variant_predictions = {}
                for variant in MODEL_VARIANTS:
                    model_info = task_data["models"][variant]
                    raw_values = pd.DataFrame([model_features])
                    values = transform_variant(raw_values, model_info["transformer"])
                    estimator = model_info["classifier"]
                    raw_probabilities = estimator.predict_proba(values)
                    calibrated = apply_fov_calibrators(
                        raw_probabilities, [fov], model_info["classes"],
                        model_info["probability_calibrators"],
                    )[0]
                    threshold = model_info["thresholds_by_fov"][fov]
                    dr_index = model_info["classes"].index(1)
                    is_dr = calibrated[dr_index] >= threshold
                    if task == "binary":
                        predicted_index = 1 if is_dr else 0
                    elif is_dr:
                        predicted_index = 1
                    else:
                        non_dr = [i for i, label in enumerate(model_info["classes"]) if label != 1]
                        predicted_index = model_info["classes"][non_dr[int(np.argmax(calibrated[non_dr]))]]
                    probabilities_by_class = {
                        class_labels[task][int(class_id)]: float(probability)
                        for class_id, probability in zip(model_info["classes"], calibrated)
                    }
                    predicted_name = class_labels[task][predicted_index]
                    prediction = {
                        "predicted_class": predicted_name,
                        "confidence": float(probabilities_by_class[predicted_name]),
                        "probabilities": probabilities_by_class,
                        "field_of_view_mm": fov_mm,
                        "dr_threshold": float(threshold),
                    }
                    if variant == "hybrid":
                        names = model_info["transformed_feature_names"]
                        importances = estimator.feature_importances_
                        prediction["top_topological_biomarkers"] = [
                            {"name": names[i], "importance": float(importances[i])}
                            for i in np.argsort(importances)[::-1]
                            if names[i] in TOPO_NAMES
                        ][:5]
                    variant_predictions[variant] = prediction
                hybrid_prediction = variant_predictions["hybrid"]
                classification[task] = {
                    **hybrid_prediction,
                    "selected_model": "hybrid",
                    "models": variant_predictions,
                }

        original_url = f"/results/{image_id}_original.png"
        mask_url = f"/results/{image_id}_segmentation.png"
        skeleton_url = f"/results/{image_id}_skeleton.png"

        Image.fromarray((image * 255).astype(np.uint8)).save(
            OUTPUT / f"{image_id}_original.png"
        )
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            OUTPUT / f"{image_id}_segmentation.png"
        )
        Image.fromarray((skeleton.astype(np.uint8) * 255)).save(
            OUTPUT / f"{image_id}_skeleton.png"
        )

        return {
            "success": True,
            "image_id": image_id,
            "images": {
                "original": original_url,
                "segmentation": mask_url,
                "skeleton": skeleton_url
            },
            "vessel_density_percent": float(mask.mean() * 100),
            "skeleton_pixels": int(skeleton.sum()),
            "topology_features": measurements,
            "classification": classification,
            "fov": fov,
            "note": "Research image analysis only; not a medical diagnosis."
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, f"Image processing failed: {str(error)}")
