from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from fastapi.responses import HTMLResponse

app = FastAPI()

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"


class StudentData(BaseModel):
    attendance: float
    study_hours: float
    previous_grade: float
    course: str


def load_and_prepare_data():
    df = pd.read_csv("C:/Users/Дмитрий/PycharmProjects/ML_tests/students.csv")
    df = df.dropna()
    le = LabelEncoder()
    df['course_encoded'] = le.fit_transform(df['course'])
    features = ['attendance', 'study_hours', 'previous_grade', 'course_encoded']
    X = df[features]
    y = df['pass']
    return X, y, le


def train_model():
    X, y, le = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    return accuracy


@app.on_event("startup")
def startup():
    if not os.path.exists(MODEL_PATH):
        train_model()


@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Student Assistant</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.9);
                padding: 30px;
                border-radius: 15px;
                color: #333;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                text-align: center;
                color: #667eea;
            }
            input, select {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            button {
                width: 100%;
                padding: 12px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 18px;
                cursor: pointer;
                margin-top: 20px;
            }
            button:hover {
                background: #764ba2;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
            }
            .pass {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .fail {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .nav-buttons {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .nav-btn {
                background: #667eea;
                color: white;
                text-decoration: none;
                padding: 10px 15px;
                border-radius: 5px;
                text-align: center;
                flex: 1;
            }
            .nav-btn:hover {
                background: #764ba2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav-buttons">
                <a href="/" class="nav-btn">Prediction</a>
                <a href="/viz" class="nav-btn">Charts</a>
            </div>
            <h1>Smart Student Assistant</h1>
            <h3>Student Performance Prediction</h3>

            <form id="predictForm">
                <label>Attendance (%):</label>
                <input type="number" id="attendance" required step="any">

                <label>Study hours per week:</label>
                <input type="number" id="study_hours" required step="any">

                <label>Previous grade:</label>
                <input type="number" id="previous_grade" required step="any">

                <label>Course:</label>
                <select id="course">
                    <option value="Math">Math</option>
                    <option value="Physics">Physics</option>
                </select>

                <button type="submit">Get Prediction</button>
            </form>

            <div class="loading" id="loading">
                <p>Analyzing data...</p>
            </div>

            <div id="result"></div>
        </div>

        <script>
            document.getElementById('predictForm').onsubmit = async (e) => {
                e.preventDefault();
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';

                const data = {
                    attendance: parseFloat(document.getElementById('attendance').value),
                    study_hours: parseFloat(document.getElementById('study_hours').value),
                    previous_grade: parseFloat(document.getElementById('previous_grade').value),
                    course: document.getElementById('course').value
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                document.getElementById('loading').style.display = 'none';

                const resultDiv = document.getElementById('result');
                if (result.will_pass) {
                    resultDiv.innerHTML = `
                        <div class="result pass">
                            Student WILL PASS the exam
                            Success probability: ${(result.probability * 100).toFixed(1)}%
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result fail">
                            Student WILL NOT PASS the exam
                            Success probability: ${(result.probability * 100).toFixed(1)}%
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/train")
async def train():
    accuracy = train_model()
    return {"status": "success", "accuracy": accuracy}


@app.post("/predict")
async def predict(student: StudentData):
    if not os.path.exists(MODEL_PATH):
        train_model()

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    course_encoded = le.transform([student.course])[0]
    input_data = np.array([[student.attendance, student.study_hours, student.previous_grade, course_encoded]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {"will_pass": bool(prediction), "probability": float(probability)}


@app.get("/data")
async def get_data():
    df = pd.read_csv("C:/Users/Дмитрий/PycharmProjects/ML_tests/students.csv")
    return df.to_dict(orient="records")


@app.get("/viz", response_class=HTMLResponse)
async def viz():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Student Data Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                text-align: center;
                color: #667eea;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .nav-buttons {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
            }
            .nav-btn {
                background: #667eea;
                color: white;
                text-decoration: none;
                padding: 10px 15px;
                border-radius: 5px;
                text-align: center;
                flex: 1;
            }
            .nav-btn:hover {
                background: #764ba2;
            }
            .stats {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .stats h3 {
                color: #667eea;
                margin-top: 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav-buttons">
                <a href="/" class="nav-btn">Prediction</a>
                <a href="/viz" class="nav-btn">Charts</a>
            </div>
            <h1>Student Performance Analysis</h1>
            <div class="subtitle">Hover over bars to see exact values</div>
            <div id="plot1" style="height: 500px; margin-bottom: 30px;"></div>
            <div id="plot2" style="height: 500px; margin-bottom: 30px;"></div>
            <div id="plot3" style="height: 500px;"></div>
            <div class="stats" id="stats"></div>
        </div>

        <script>
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    const attendance = data.map(d => d.attendance);
                    const studyHours = data.map(d => d.study_hours);
                    const grades = data.map(d => d.previous_grade);

                    const trace1 = {
                        x: attendance,
                        type: 'histogram',
                        name: 'Attendance',
                        opacity: 0.7,
                        marker: { color: '#667eea', line: { color: '#fff', width: 1 } },
                        hovertemplate: 'Value: %{x}<br>Count: %{y}<extra></extra>',
                        nbinsx: 8
                    };

                    const trace2 = {
                        x: studyHours,
                        type: 'histogram',
                        name: 'Study Hours',
                        opacity: 0.7,
                        marker: { color: '#764ba2', line: { color: '#fff', width: 1 } },
                        hovertemplate: 'Value: %{x} hours<br>Count: %{y}<extra></extra>',
                        nbinsx: 8
                    };

                    const trace3 = {
                        x: grades,
                        type: 'histogram',
                        name: 'Previous Grades',
                        opacity: 0.7,
                        marker: { color: '#ff6b6b', line: { color: '#fff', width: 1 } },
                        hovertemplate: 'Value: %{x}<br>Count: %{y}<extra></extra>',
                        nbinsx: 8
                    };

                    const layout1 = {
                        title: { text: 'Student Attendance Distribution', font: { size: 18, family: 'Arial' } },
                        xaxis: { title: 'Attendance (%)', gridcolor: '#eee' },
                        yaxis: { title: 'Number of Students', gridcolor: '#eee' },
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white',
                        bargap: 0.1,
                        hovermode: 'closest'
                    };

                    const layout2 = {
                        title: { text: 'Weekly Study Hours Distribution', font: { size: 18, family: 'Arial' } },
                        xaxis: { title: 'Study Hours', gridcolor: '#eee' },
                        yaxis: { title: 'Number of Students', gridcolor: '#eee' },
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white',
                        bargap: 0.1,
                        hovermode: 'closest'
                    };

                    const layout3 = {
                        title: { text: 'Previous Grades Distribution', font: { size: 18, family: 'Arial' } },
                        xaxis: { title: 'Average Grade', gridcolor: '#eee' },
                        yaxis: { title: 'Number of Students', gridcolor: '#eee' },
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white',
                        bargap: 0.1,
                        hovermode: 'closest'
                    };

                    Plotly.newPlot('plot1', [trace1], layout1);
                    Plotly.newPlot('plot2', [trace2], layout2);
                    Plotly.newPlot('plot3', [trace3], layout3);

                    const passCount = data.filter(d => d.pass === 1).length;
                    const failCount = data.filter(d => d.pass === 0).length;
                    const avgAttendance = (attendance.reduce((a,b) => a + b, 0) / attendance.length).toFixed(1);
                    const avgStudyHours = (studyHours.reduce((a,b) => a + b, 0) / studyHours.length).toFixed(1);
                    const avgGrade = (grades.reduce((a,b) => a + b, 0) / grades.length).toFixed(1);

                    document.getElementById('stats').innerHTML = `
                        <h3>Data Statistics</h3>
                        <p><strong>Total Students:</strong> ${data.length}</p>
                        <p><strong>Passed Exam:</strong> ${passCount} (${(passCount/data.length*100).toFixed(1)}%)</p>
                        <p><strong>Failed Exam:</strong> ${failCount} (${(failCount/data.length*100).toFixed(1)}%)</p>
                        <p><strong>Average Attendance:</strong> ${avgAttendance}%</p>
                        <p><strong>Average Study Hours:</strong> ${avgStudyHours} hours per week</p>
                        <p><strong>Average Previous Grade:</strong> ${avgGrade}</p>
                    `;
                });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 1. Install required libraries (run in terminal):
#    pip install fastapi uvicorn pandas numpy scikit-learn
#
# 2. Start the server (in project folder):
#    uvicorn exam1:app --reload
#
# 3. Open in browser:
#    http://localhost:8000 - main page with prediction form
#    http://localhost:8000/viz - charts page
#    http://localhost:8000/docs - API documentation
#
# 4. To stop the server: press Ctrl+C in terminal