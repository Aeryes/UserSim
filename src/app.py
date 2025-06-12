from flask import Flask
from routes.routes import dashboard

flask_app = Flask(__name__)

def create_app():

    # Register blueprints
    flask_app.register_blueprint(dashboard, url_prefix="/v1")

    return flask_app
