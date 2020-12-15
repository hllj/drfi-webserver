from flask import (
    Flask,
    request,
    url_for,
    redirect,
    render_template,
    flash,
    send_from_directory,
)
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])
CUR_FOLDER = os.getcwd()
UPLOAD_FOLDER = os.path.join(CUR_FOLDER, "uploaded")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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
            return redirect(url_for("show", filename=filename))
    else:
        return render_template("home.html")


@app.route("/uploads/<path:filename>")
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


@app.route("/photo/<filename>")
def show(filename):
    return render_template("show.html", filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
