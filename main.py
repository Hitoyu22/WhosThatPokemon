from flask import Flask

class App:
    """
    Classe principale de l'application
    """
    def __init__(self):
        self.app = Flask(__name__)

        self.configure_routes()

    def configure_routes(self):
        """
        Méthode pour configurer les routes de l'application
        """
        from app.routes import app
        self.app = app

    def run(self):
        """
        Méthode pour lancer l'application
        """
        self.app.run(debug=True)

if __name__ == "__main__":
    application = App()
    application.run()