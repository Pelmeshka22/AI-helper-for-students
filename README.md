HOW TO USE THIS APPLICATION:

Start the server:
Run this command in terminal: uvicorn exam1:app --reload

Open your browser and go to:
http://localhost:8000

Fill in the form with student data:

Attendance (percentage from 0 to 100)

Study hours per week (hours spent studying)

Previous grade (average grade from 0 to 100)

Course (select Math or Physics)

Click "Get Prediction" button

Read the result:

Green message "Student WILL PASS the exam" - student will likely succeed

Red message "Student WILL NOT PASS the exam" - student is at risk of failing

The probability percentage shows how confident the AI is

To view charts and statistics:
Open http://localhost:8000/viz

To see API documentation:
Open http://localhost:8000/docs

To stop the server:
Press Ctrl + C in the terminal

Example input:

Attendance: 85

Study hours: 6

Previous grade: 78

Course: Math

Result: Student WILL PASS the exam (87.5% probability)

The AI model learns from the students.csv file and predicts exam success based on past student
