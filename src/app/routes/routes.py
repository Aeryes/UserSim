from flask import Flask, render_template, request, redirect, url_for, session
import os

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../templates"))
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../static"))
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

from app.routes.training_dashboard import admin_bp
app.register_blueprint(admin_bp)

app.secret_key = 'secret_key'
users = {}
import os
print("Template folder:", os.path.abspath("templates"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        users[email] = password
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if users.get(email) == password:
            session['user'] = email
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])
