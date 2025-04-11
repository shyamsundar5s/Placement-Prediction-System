from flask import Flask, render_template, request
import sqlite3
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def log_to_db(data, result):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (cgpa REAL, internships INTEGER, projects INTEGER, extracurricular INTEGER, prediction TEXT)''')
    c.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", (*data, result))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cgpa = float(request.form['cgpa'])
        internships = int(request.form['internships'])
        projects = int(request.form['projects'])
        extracurricular = int(request.form['extracurricular'])

        input_data = np.array([[cgpa, internships, projects, extracurricular]])
        prediction = model.predict(input_data)[0]
        result = 'Placed' if prediction == 1 else 'Not Placed'

        log_to_db([cgpa, internships, projects, extracurricular], result)
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
