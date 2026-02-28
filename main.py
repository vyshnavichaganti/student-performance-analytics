from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Student Performance Prediction System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model/student_performance_model.pkl")

CAT_COLS = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
            'SectionID', 'Topic', 'Semester', 'Relation',
            'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']

class StudentInput(BaseModel):
    gender: str
    NationalITy: str
    PlaceofBirth: str
    StageID: str
    GradeID: str
    SectionID: str
    Topic: str
    Semester: str
    Relation: str
    raisedhands: int
    VisITedResources: int
    AnnouncementsView: int
    Discussion: int
    ParentAnsweringSurvey: str
    ParentschoolSatisfaction: str
    StudentAbsenceDays: str

class PredictionResult(BaseModel):
    prediction: int
    label: str
    confidence: float
    risk_level: str
    recommendations: list

@app.post("/predict", response_model=PredictionResult)
def predict(student: StudentInput):
    input_df = pd.DataFrame([student.dict()])

    # Encode categorical columns
    le = LabelEncoder()
    for col in CAT_COLS:
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(float(max(proba)) * 100, 2)

    labels = {0: "Low Performance", 1: "Medium Performance", 2: "High Performance"}
    risks = {0: "High Risk", 1: "Medium Risk", 2: "Low Risk"}
    label = labels[prediction]
    risk = risks[prediction]

    recommendations = []
    if student.raisedhands < 30:
        recommendations.append("Encourage more class participation — raise hands more often!")
    if student.VisITedResources < 40:
        recommendations.append("Visit more learning resources and study materials!")
    if student.AnnouncementsView < 30:
        recommendations.append("Check announcements regularly to stay updated!")
    if student.Discussion < 20:
        recommendations.append("Participate more in discussion forums!")
    if student.StudentAbsenceDays == "Above-7":
        recommendations.append("Reduce absences — attendance impacts performance significantly!")
    if not recommendations:
        recommendations.append("Great performance! Keep up the excellent work!")

    return PredictionResult(
        prediction=int(prediction),
        label=label,
        confidence=confidence,
        risk_level=risk,
        recommendations=recommendations
    )

@app.get("/health")
def health():
    return {"status": "running"}