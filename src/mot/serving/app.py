from flask import Flask, render_template, request

from mot.serving.inference import handle_post_request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return handle_post_request()
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(threaded=True, port=5000, debug=False, host="0.0.0.0")
