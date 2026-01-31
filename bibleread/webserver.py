# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify
import json, os
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

STATUS_FILE = "scheduler_status.json"
LOG_FILE    = "scheduler.log"

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route("/api/logs")
def api_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return jsonify({"logs": f.readlines()[-50:]})
    return jsonify({"logs": []})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5003)
