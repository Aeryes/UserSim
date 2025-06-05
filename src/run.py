from app.routes.routes import app

if __name__ == '__main__':
    from init_files import *  # or just call the logic directly in run.py
    app.run(debug=True, use_reloader=False)

