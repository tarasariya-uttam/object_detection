from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Congratulations Uttamkumar, it's a web app!"
