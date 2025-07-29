from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import uvicorn

app = FastAPI(title="Lung Cancer Risk Predictor API")

# Global model variable
model = None
feature_names = [
    'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',  
    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
    'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'GENDER_encoded'
]

# Pydantic model for request data
class PredictionRequest(BaseModel):
    gender: str
    age: int
    smoking: int
    alcohol: int
    yellow_fingers: int
    anxiety: int
    peer_pressure: int
    chronic_disease: int
    fatigue: int
    allergy: int
    wheezing: int
    coughing: int
    shortness_breath: int
    swallowing_difficulty: int
    chest_pain: int

def load_and_train_model():
    global model
    try:
        # Load data
        data = pd.read_csv('survey lung cancer (2).csv')
        
        # Preprocess data
        df = data.copy()
        df['GENDER_encoded'] = df['GENDER'].map({'M': 1, 'F': 0})
        df['LUNG_CANCER_encoded'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        # Features and target
        X = df[feature_names]
        y = df['LUNG_CANCER_encoded']
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        print("Model trained successfully!")
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    success = load_and_train_model()
    if not success:
        print("Warning: Model failed to load")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lung Cancer Risk Predictor</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 800px; margin: 50px auto; }
            .card { border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
            .btn-predict { background: linear-gradient(45deg, #667eea, #764ba2); border: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="card-header text-center bg-primary text-white">
                    <h1>🫁 Lung Cancer Risk Predictor</h1>
                    <p>FastAPI + Machine Learning</p>
                </div>
                <div class="card-body p-4">
                    <form id="predictionForm">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <label class="form-label">Gender</label>
                                <select class="form-select" id="gender" required>
                                    <option value="">Select Gender</option>
                                    <option value="male">Male</option>
                                    <option value="female">Female</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Age</label>
                                <input type="number" class="form-control" id="age" min="20" max="90" required>
                            </div>
                        </div>
                        
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <label class="form-label">Smoking (1=No, 2=Yes)</label>
                                <select class="form-select" id="smoking" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Alcohol (1=No, 2=Yes)</label>
                                <select class="form-select" id="alcohol" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                        </div>
                        
                        <!-- Add more symptom fields -->
                        <div class="row mb-3">
                            <div class="col-md-4">
                                <label class="form-label">Yellow Fingers</label>
                                <select class="form-select" id="yellow_fingers" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Anxiety</label>
                                <select class="form-select" id="anxiety" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Chronic Disease</label>
                                <select class="form-select" id="chronic_disease" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="row mb-3">
                            <div class="col-md-4">
                                <label class="form-label">Fatigue</label>
                                <select class="form-select" id="fatigue" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Allergy</label>
                                <select class="form-select" id="allergy" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Wheezing</label>
                                <select class="form-select" id="wheezing" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="row mb-3">
                            <div class="col-md-4">
                                <label class="form-label">Coughing</label>
                                <select class="form-select" id="coughing" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Shortness of Breath</label>
                                <select class="form-select" id="shortness_breath" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Chest Pain</label>
                                <select class="form-select" id="chest_pain" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <label class="form-label">Peer Pressure</label>
                                <select class="form-select" id="peer_pressure" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Swallowing Difficulty</label>
                                <select class="form-select" id="swallowing_difficulty" required>
                                    <option value="">Select</option>
                                    <option value="1">No</option>
                                    <option value="2">Yes</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="text-center">
                            <button type="submit" class="btn btn-predict btn-lg text-white px-5">
                                🔍 Predict Risk
                            </button>
                        </div>
                    </form>
                    
                    <div id="result" class="mt-4" style="display: none;"></div>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = {
                    gender: document.getElementById('gender').value,
                    age: parseInt(document.getElementById('age').value),
                    smoking: parseInt(document.getElementById('smoking').value),
                    alcohol: parseInt(document.getElementById('alcohol').value),
                    yellow_fingers: parseInt(document.getElementById('yellow_fingers').value),
                    anxiety: parseInt(document.getElementById('anxiety').value),
                    peer_pressure: parseInt(document.getElementById('peer_pressure').value),
                    chronic_disease: parseInt(document.getElementById('chronic_disease').value),
                    fatigue: parseInt(document.getElementById('fatigue').value),
                    allergy: parseInt(document.getElementById('allergy').value),
                    wheezing: parseInt(document.getElementById('wheezing').value),
                    coughing: parseInt(document.getElementById('coughing').value),
                    shortness_breath: parseInt(document.getElementById('shortness_breath').value),
                    swallowing_difficulty: parseInt(document.getElementById('swallowing_difficulty').value),
                    chest_pain: parseInt(document.getElementById('chest_pain').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    
                    const resultDiv = document.getElementById('result');
                    const riskClass = result.risk_level === 'HIGH' ? 'alert-danger' : 'alert-success';
                    const probability = (result.probability.cancer * 100).toFixed(1);
                    
                    resultDiv.innerHTML = `
                        <div class="alert ${riskClass} text-center">
                            <h3>${result.risk_level} RISK</h3>
                            <p class="mb-0">Cancer Risk Probability: <strong>${probability}%</strong></p>
                        </div>
                    `;
                    
                    resultDiv.style.display = 'block';
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict_cancer_risk(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert gender to numeric
        gender_encoded = 1 if request.gender.lower() == 'male' else 0
        
        # Prepare feature array
        features = [
            request.age,
            request.smoking,
            request.yellow_fingers,
            request.anxiety,
            request.peer_pressure,
            request.chronic_disease,
            request.fatigue,
            request.allergy,
            request.wheezing,
            request.alcohol,
            request.coughing,
            request.shortness_breath,
            request.swallowing_difficulty,
            request.chest_pain,
            gender_encoded
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        return {
            "prediction": int(prediction),
            "probability": {
                "no_cancer": float(probability[0]),
                "cancer": float(probability[1])
            },
            "risk_level": "HIGH" if prediction == 1 else "LOW"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)