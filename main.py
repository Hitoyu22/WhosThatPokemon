from flask import Flask


class App:
    def __init__(self):
        self.app = Flask(__name__)

        self.configure_routes()

    def configure_routes(self):
        from app.routes import app
        self.app = app

    def run(self):
        self.app.run(debug=True)

if __name__ == "__main__":
    application = App()
    application.run()