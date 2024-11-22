# Code by AkinoAlice@TyrantRey

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils.vector_extractor import VectorExtractor
from utils.ocr import OCR

import os


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
load_dotenv()

OCR_SCANNER = OCR()
UPLOAD_FOLDER = "./uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """helper function to check if a file allowed

    Args:
        filename (str): filename to check

    Returns:
        bool: True if the file is allowed
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index() -> str:
    """Rendering the index.html

    Returns:
        str: index.html
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """uploading file

    Returns:
        Response: request url
    """
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER + filename
            file.save(filepath)

            # Perform OCR on the uploaded file
            ocr_result = OCR_SCANNER.scan(filepath)

        else:
            flash("Allowed file types are png, jpg, jpeg, gif")
            return redirect(request.url)

    return render_template("index.html", ocr_result=ocr_result)


if __name__ == "__main__":
    app.run(debug=True)
