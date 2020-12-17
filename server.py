from flask import (
    Flask,
    request,
    url_for,
    redirect,
    render_template,
    send_from_directory,
)
import os
from drfi.test import main

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])
CUR_FOLDER = os.getcwd()
UPLOAD_FOLDER = os.path.join(CUR_FOLDER, "uploaded")
OUTPUT_FOLDER = os.path.join(CUR_FOLDER, "drfi/data/result")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# model constants
MODEL_PATH_SAME_REGION = os.path.join(CUR_FOLDER, "drfi/data/model/rf_same_region.pkl")
MODEL_PATH_SALIENCE = os.path.join(CUR_FOLDER, "drfi/data/model/rf_salience.pkl")
MODEL_PATH_FUSION_MLP = os.path.join(CUR_FOLDER, "drfi/data/model/mlp.pkl")
MODEL_PATH_FUSION_XGB = os.path.join(CUR_FOLDER, "drfi/data/model/xgb.pkl")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["photo"]
        if file and allowed_file(file.filename):
            filename = file.filename
            save_filename = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_filename)
            result_filename, result_path, result_img = main(
                img_path=save_filename,
                threshold=1,
                fm="mlp",
                model_path_same_region=MODEL_PATH_SAME_REGION,
                model_path_salience=MODEL_PATH_SALIENCE,
                model_path_fusion=MODEL_PATH_FUSION_MLP,
            )
            print(result_filename, result_path)

            return redirect(
                url_for("show", input_img=filename, output_img=result_filename)
            )
    else:
        return render_template("home.html")


@app.route("/input/<path:filename>")
def download_upload_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


@app.route("/output/<path:filename>")
def download_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route("/photo/<input_img>/<output_img>")
def show(input_img, output_img):
    return render_template("show.html", input_img=input_img, output_img=output_img)


if __name__ == "__main__":
    app.run(debug=True)
