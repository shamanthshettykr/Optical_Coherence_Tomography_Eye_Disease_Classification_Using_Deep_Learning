import os, uuid, json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from database import init_db, SessionLocal, PredictionHistory
from ml_utils import predict_disease

app = Flask(__name__)
app.secret_key = "oculoai_secret_2026"

UPLOAD_FOLDER     = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Home ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Information page ──────────────────────────────────────────────────────────
@app.route("/information")
def information():
    return render_template("information.html")


# ── Prediction page (upload + inference) ─────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            ext      = file.filename.rsplit(".", 1)[1].lower()
            filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result = predict_disease(filepath)

            db = SessionLocal()
            try:
                db.add(PredictionHistory(
                    filename=filename,
                    predicted_class=result["class"],
                    confidence=result["confidence"],
                ))
                db.commit()
            finally:
                db.close()

            return render_template(
                "result.html",
                filename=filename,
                prediction=result["class"],
                short_class=result["short_class"],
                confidence=round(result["confidence"] * 100, 2),
                description=result["description"],
                precautions=result["precautions"],
            )
        else:
            flash("Allowed file types: png, jpg, jpeg")
            return redirect(request.url)

    return render_template("predict.html")


# ── Result analysis (model metrics dashboard) ─────────────────────────────────
@app.route("/analysis")
def analysis():
    metrics_path = os.path.join("static", "data", "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return render_template("analysis.html", metrics=metrics)


# ── History ───────────────────────────────────────────────────────────────────
@app.route("/history")
def history():
    db = SessionLocal()
    try:
        records = (db.query(PredictionHistory)
                   .order_by(PredictionHistory.timestamp.desc()).all())
    finally:
        db.close()
    return render_template("history.html", records=records)


# ── Legacy redirect: old "/" POST still works ─────────────────────────────────
@app.route("/legacy", methods=["POST"])
def legacy():
    return redirect(url_for("predict"), code=307)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
