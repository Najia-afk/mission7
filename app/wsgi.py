# app/wsgi.py
"""
WSGI entry point for production deployment with Gunicorn.
Usage: gunicorn app.wsgi:app
"""
from app.main import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
