from flask import (
    Flask,
    request,
    url_for,
    redirect,
    render_template,
    send_from_directory,
)
import os
from test import predict
import cv2

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])
CUR_FOLDER = os.getcwd()
UPLOAD_FOLDER = os.path.join(CUR_FOLDER, "uploaded")
OUTPUT_FOLDER = os.path.join(CUR_FOLDER, "data/result")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# model constants
CONFIG_PATH = os.path.join(CUR_FOLDER, "config/config.yaml")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


def preprocess_resize(img_path):
    img = cv2.imread(img_path)
    original_height, original_width = img.shape[0], img.shape[1]
    is_resize = False
    if original_height > 400 or original_width > 400:
        is_resize = True
        if original_height > original_width:
            ratio = 400.0 / original_height
        else:
            ratio = 400.0 / original_width
        resize_width = int(original_width * ratio)
        resize_height = int(original_height * ratio)
        dim = (resize_width, resize_height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, img)
    print(is_resize, original_height, original_width)
    return is_resize, original_height, original_width


def upsize_postprocess(is_resize, original_height, original_width, img_path):
    if is_resize:
        img = cv2.imread(img_path)
        dim = (original_width, original_height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path, img)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["photo"]
        if file and allowed_file(file.filename):
            filename = file.filename
            save_filename = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_filename)
            is_resize, original_height, original_width = preprocess_resize(
                save_filename
            )
            mask, result_img, result_filename, result_path = predict(
                CONFIG_PATH, save_filename
            )
            print(result_filename, result_path)
            upsize_postprocess(is_resize, original_height, original_width, result_path)
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
