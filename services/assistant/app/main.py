import os
import pickle
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    # Create type hints for when imports are available
    _AutoTokenizer = AutoTokenizer
    _AutoModelForSequenceClassification = AutoModelForSequenceClassification
    _torch = torch
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. NLP email urgency detection will be disabled.")
    # Create placeholder types for when imports aren't available
    _AutoTokenizer = None
    _AutoModelForSequenceClassification = None
    _torch = None

# --- Decision Engine Class Definition (Required for Pickle Loading) ---
class DecisionEngine:
    """
    Decision Engine class definition needed for pickle loading.
    This matches the class structure from your Jupyter notebook.
    """
    def __init__(self, nlp_model_path=None, stress_model_path=None):
        self.nlp_model_path = nlp_model_path
        self.stress_model_path = stress_model_path
        self.stress_model = None
        # Updated feature order to match your actual ML model from the notebook
        self.feature_order = ["Sleep_Duration", "Calendar_Busy_Hours", "HeartRate_Avg", "Steps_Last_24h", "Urgent_Emails_Flag"]
    
    def __call__(self, email_text, stress_features):
        # This method will be overridden by the pickled object
        pass
    
    def recommend_break(self, features):
        # This method will be overridden by the pickled object
        return "Take a break"
    
    def predict(self, features_df):
        # This method will be overridden by the pickled object
        # Fallback to return a default stress level if not implemented
        return [5]  # Medium stress level

# --- Service Configuration ---
# Service URLs for local development (use localhost instead of docker service names)
INTEGRATIONS_SERVICE_URL = os.getenv("INTEGRATIONS_SERVICE_URL", "http://localhost:8001")
ACTIONS_SERVICE_URL = os.getenv("ACTIONS_SERVICE_URL", "http://localhost:8003")

# --- Model Loading ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "ml_models", "decision_engine.pkl")

# Custom unpickler to map notebook '__main__.DecisionEngine' to our runtime class
class _RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "DecisionEngine":
            # Defer import to runtime to avoid import cycles
            from app.engine_runtime import DecisionEngine as RuntimeDecisionEngine  # type: ignore
            return RuntimeDecisionEngine
        return super().find_class(module, name)

DECISION_ENGINE = None
try:
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    with open(MODEL_FILE, 'rb') as f:
        DECISION_ENGINE = _RenameUnpickler(f).load()
    print(f"Successfully loaded Decision Engine via remapped unpickler: {MODEL_FILE}")
except FileNotFoundError:
    print(f"Model file not found at {MODEL_FILE}. NOT creating fallback. Please ensure the model exists.")
except Exception as e:
    print(f"Error loading ML model via remapped unpickler: {e}. NOT creating fallback.")

app = FastAPI(
    title="Harmonia Assistant Service",
    description="AI-powered holistic assistant that analyzes user data and provides personalized wellness recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Pydantic Models for Request/Response Validation ---

class RecommendationRequest(BaseModel):
    """
    Request model for getting AI-powered wellness recommendations.
    """
    user_token: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    
    @validator('user_token')
    def validate_user_token(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_token cannot be empty')
        return v.strip()

class ActionDetails(BaseModel):
    """
    Details of the action to be performed.
    """
    title: Optional[str] = None
    duration: Optional[int] = None
    description: Optional[str] = None
    message: Optional[str] = None
    channel: Optional[str] = None
    to: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None

class FeatureData(BaseModel):
    """
    ML model features extracted from user data.
    """
    Sleep_Duration: float
    Calendar_Busy_Hours: float
    HeartRate_Avg: float
    Steps_Last_24h: float
    Urgent_Emails_Flag: float

class RecommendationResponse(BaseModel):
    """
    Response model for wellness recommendations.
    """
    status: str
    recommendation: str
    stress_level: int
    action_taken: str
    action_details: ActionDetails
    features_used: FeatureData
    action_service_response: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str
    service: str
    timestamp: str
    checks: Dict[str, Dict[str, Any]]

class ErrorResponse(BaseModel):
    """
    Error response model.
    """
    detail: str
    error_code: Optional[str] = None
    timestamp: str

# --- Action Mapping ---
# Maps ML model predictions to valid Actions Service payloads
# Based on your decision engine's stress level predictions (0-10 scale)
ACTION_MAPPING = {
    # Stress Level 0-2: Low stress, positive reinforcement
    0: {
        "action": "send_slack_notification",
        "details": {
            "message": "🌟 You're in a great flow state! Keep up the excellent work.",
            "channel": "#team-harmonia"
        },
    },
    1: {
        "action": "send_slack_notification", 
        "details": {
            "message": "💪 Great energy levels! Stay focused and productive.",
            "channel": "#team-harmonia"
        },
    },
    2: {
        "action": "send_slack_notification",
        "details": {
            "message": "✨ You're doing well! Remember to stay hydrated.",
            "channel": "#team-harmonia"
        },
    },
    
    # Stress Level 3-4: Mild stress, gentle suggestions
    3: {
        "action": "create_break_event",
        "details": {
            "title": "Quick Refresher",
            "duration": 10,
            "description": "AI suggests: Take a short breather to maintain your energy"
        },
    },
    4: {
        "action": "create_break_event",
        "details": {
            "title": "Stretch Break",
            "duration": 15,
            "description": "AI suggests: Time for a quick stretch and some deep breaths"
        },
    },
    
    # Stress Level 5-6: Moderate stress, more active interventions
    5: {
        "action": "create_break_event",
        "details": {
            "title": "Mindfulness Break",
            "duration": 20,
            "description": "AI suggests: Take time for meditation or a short walk"
        },
    },
    6: {
        "action": "send_slack_notification",
        "details": {
            "message": "🧘‍♀️ Consider taking a longer break to recharge. Your wellbeing matters!",
            "channel": "#team-harmonia"
        },
    },
    
    # Stress Level 7-8: High stress, strong recommendations
    7: {
        "action": "create_break_event",
        "details": {
            "title": "Wellness Break",
            "duration": 30,
            "description": "AI Alert: High stress detected. Time for a proper break and reset."
        },
    },
    8: {
        "action": "send_slack_notification",
        "details": {
            "message": "⚠️ High stress levels detected. Please prioritize your wellbeing and consider delegating tasks.",
            "channel": "#team-harmonia"
        },
    },
    
    # Stress Level 9-10: Very high stress, urgent interventions
    9: {
        "action": "create_break_event",
        "details": {
            "title": "URGENT: Wellness Priority",
            "duration": 60,
            "description": "AI ALERT: Critical stress levels. Please step away and focus on self-care."
        },
    },
    10: {
        "action": "draft_email",
        "details": {
            "to": "manager@company.com",
            "subject": "Wellness Check - High Stress Alert",
            "body": "AI Health Assistant Alert: High stress levels detected. May need support or workload adjustment. Please prioritize wellbeing."
        },
    }
}

# --- Feature Engineering Functions ---
def extract_features_from_raw_data(raw_user_data: Dict[str, Any], user_token: str) -> Dict[str, float]:
    """
    Transform raw API data into the 5 features required by the ML model:
    Sleep_Duration, Calendar_Busy_Hours, HeartRate_Avg, Steps_Last_24h, Urgent_Emails_Flag
    """
    calendar_events = raw_user_data.get('calendar_events', [])
    heart_rate_data = raw_user_data.get('heart_rate_data', [])
    
    # Calculate features
    calendar_busy_hours = calculate_calendar_busy_hours(calendar_events)
    heart_rate_avg = calculate_heart_rate_average(heart_rate_data)
    sleep_duration = estimate_sleep_duration(calendar_events)  # Estimate from calendar patterns
    steps_last_24h = extract_steps_from_fitness_data(heart_rate_data)  # Estimate from available data
    
    # NLP-based urgency detection
    emails = fetch_emails_for_urgency_analysis(user_token)
    urgent_emails_flag = analyze_email_urgency(emails)
    
    return {
        "Sleep_Duration": sleep_duration,
        "Calendar_Busy_Hours": calendar_busy_hours,
        "HeartRate_Avg": heart_rate_avg,
        "Steps_Last_24h": steps_last_24h,
        "Urgent_Emails_Flag": urgent_emails_flag
    }

def calculate_calendar_busy_hours(calendar_events: List[Dict]) -> float:
    """
    Calculate total busy hours from calendar events in the last 24 hours.
    """
    if not calendar_events:
        return 0.0
    
    total_busy_minutes = 0
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(hours=24)
    
    for event in calendar_events:
        try:
            # Parse event times
            start_time = None
            end_time = None
            
            if 'start' in event:
                start_str = event['start'].get('dateTime') or event['start'].get('date')
                if start_str:
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            
            if 'end' in event:
                end_str = event['end'].get('dateTime') or event['end'].get('date')
                if end_str:
                    end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
            
            # Calculate duration if both times are available and within last 24h
            if start_time and end_time and start_time >= yesterday:
                duration_minutes = (end_time - start_time).total_seconds() / 60
                total_busy_minutes += max(0, duration_minutes)
                
        except (ValueError, KeyError) as e:
            # Skip events with invalid time formats
            continue
    
    return round(total_busy_minutes / 60, 2)  # Convert to hours

def calculate_heart_rate_average(heart_rate_data: List[Dict]) -> float:
    """
    Calculate average heart rate from the last 24 hours of data.
    """
    if not heart_rate_data:
        return 70.0  # Default resting heart rate
    
    heart_rates = []
    
    for data_point in heart_rate_data:
        try:
            # Extract heart rate values from Google Fit data structure
            if 'value' in data_point:
                for value in data_point['value']:
                    if 'fpVal' in value:  # Heart rate is stored as floating point value
                        heart_rates.append(value['fpVal'])
                    elif 'intVal' in value:  # Sometimes as integer
                        heart_rates.append(float(value['intVal']))
        except (KeyError, TypeError):
            continue
    
    if heart_rates:
        return round(sum(heart_rates) / len(heart_rates), 1)
    else:
        return 70.0  # Default if no valid data

def estimate_sleep_duration(calendar_events: List[Dict]) -> float:
    """
    Estimate sleep duration based on calendar patterns and gaps.
    This is a heuristic approach since we don't have direct sleep data.
    """
    if not calendar_events:
        return 7.0  # Default 7 hours
    
    # Look for patterns in calendar events to estimate sleep
    # Find the longest gap between events during typical sleep hours (10 PM - 8 AM)
    gaps = []
    now = datetime.now(timezone.utc)
    
    # Sort events by start time
    sorted_events = []
    for event in calendar_events:
        try:
            if 'start' in event:
                start_str = event['start'].get('dateTime')
                if start_str:
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    sorted_events.append(start_time)
        except (ValueError, KeyError):
            continue
    
    sorted_events.sort()
    
    # Calculate gaps between consecutive events
    for i in range(1, len(sorted_events)):
        gap_hours = (sorted_events[i] - sorted_events[i-1]).total_seconds() / 3600
        # Consider gaps between 6-12 hours as potential sleep periods
        if 6 <= gap_hours <= 12:
            gaps.append(gap_hours)
    
    if gaps:
        return round(max(gaps), 1)  # Return the longest reasonable gap
    else:
        return 7.0  # Default

def extract_steps_from_fitness_data(heart_rate_data: List[Dict]) -> float:
    """
    Estimate daily steps from available fitness data.
    This is a placeholder - ideally we'd have dedicated step data.
    """
    # For now, estimate based on heart rate patterns
    # Higher average heart rate might indicate more activity
    avg_hr = calculate_heart_rate_average(heart_rate_data)
    
    # Simple heuristic: map heart rate to estimated steps
    if avg_hr > 80:
        return 8000.0  # Active day
    elif avg_hr > 70:
        return 5000.0  # Moderate activity
    else:
        return 2000.0  # Low activity

def fetch_emails_for_urgency_analysis(user_token: str) -> List[str]:
    """
    Fetch recent emails from Gmail API for urgency analysis.
    Returns a list of email text (subject + body) for NLP processing.
    """
    try:
        headers = {"Authorization": f"Bearer {user_token}"}
        
        # For now, we'll extract email information from calendar events
        # In a full implementation, you'd call Gmail API directly
        response = requests.get(
            f"{INTEGRATIONS_SERVICE_URL}/api/v1/data/calendar",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        calendar_data = response.json()
        
        emails = []
        # Extract email-like content from calendar event descriptions
        for event in calendar_data.get('events', []):
            summary = event.get('summary', '')
            description = event.get('description', '')
            
            # Look for email-like patterns in calendar events
            if any(keyword in (summary + ' ' + description).lower() 
                   for keyword in ['email', 'message', 'urgent', 'asap', 'important']):
                email_text = f"SUBJECT: {summary} BODY: {description}"
                emails.append(email_text)
        
        return emails[:5]  # Limit to 5 recent emails for processing
        
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def analyze_email_urgency(emails: List[str]) -> float:
    """
    Use the trained DistilBERT model to analyze email urgency.
    Returns 1.0 if urgent emails detected, 0.0 otherwise.
    """
    if not emails or not TRANSFORMERS_AVAILABLE or _torch is None:
        return analyze_email_urgency_keywords(emails)
    
    try:
        # Load the trained NLP model
        model_path = os.path.join(SCRIPT_DIR, "ml_models", "nlp_email_model")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Warning: NLP model not found at {model_path}. Using keyword-based urgency detection.")
            return analyze_email_urgency_keywords(emails)
        
        if _AutoTokenizer is None or _AutoModelForSequenceClassification is None:
            print("Warning: Transformers not available. Using keyword-based urgency detection.")
            return analyze_email_urgency_keywords(emails)
            
        tokenizer = _AutoTokenizer.from_pretrained(model_path)
        model = _AutoModelForSequenceClassification.from_pretrained(model_path)
        
        urgent_count = 0
        total_emails = len(emails)
        
        for email_text in emails:
            # Tokenize and predict
            inputs = tokenizer(email_text, 
                             padding=True, 
                             truncation=True, 
                             max_length=512,
                             return_tensors="pt")
            
            with _torch.no_grad():
                outputs = model(**inputs)
                predictions = _torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Assuming label 1 is "urgent" (based on training)
                if predictions[0][1].item() > 0.7:  # High confidence threshold
                    urgent_count += 1
        
        # Return 1.0 if any urgent emails found, 0.0 otherwise
        return 1.0 if urgent_count > 0 else 0.0
        
    except Exception as e:
        print(f"Error in NLP urgency analysis: {e}")
        # Fallback to keyword-based detection
        return analyze_email_urgency_keywords(emails)

def analyze_email_urgency_keywords(emails: List[str]) -> float:
    """
    Fallback keyword-based urgency detection.
    """
    urgency_keywords = [
        'urgent', 'asap', 'immediately', 'important', 'priority',
        'deadline', 'critical', 'emergency', 'attention', 'action required'
    ]
    
    for email in emails:
        email_lower = email.lower()
        if any(keyword in email_lower for keyword in urgency_keywords):
            return 1.0
    
    return 0.0

# --- Endpoint for Recommendation ---
@app.post(
    "/api/v1/recommend",
    response_model=RecommendationResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        503: {"model": ErrorResponse, "description": "ML Model not available"},
        502: {"model": ErrorResponse, "description": "Upstream service error"}
    },
    summary="Get AI Wellness Recommendation",
    description="Analyzes user data and provides personalized wellness recommendations based on stress levels and activity patterns."
)
async def get_recommendation(request: RecommendationRequest):
    """
    1. Fetches real-time data from the Integrations Service.
    2. Runs the Decision Engine (ML Model) to get a prediction.
    3. Dispatches the resulting action to the Actions Service.
    """
    if not DECISION_ENGINE:
        raise HTTPException(status_code=503, detail="ML Model not loaded.")

    try:
        # 1. FETCH DATA from INTEGRATIONS SERVICE
        # Extract validated data from request model
        user_token = request.user_token

        headers = {"Authorization": f"Bearer {user_token}"}
        
        # Call the aggregate endpoint for all user data
        data_response = requests.get(
            f"{INTEGRATIONS_SERVICE_URL}/api/v1/data/aggregate",
            headers=headers
        )
        data_response.raise_for_status()
        raw_user_data = data_response.json()

        # 2. FEATURE ENGINEERING
        # Transform raw API data into ML model features
        model_input_data = extract_features_from_raw_data(raw_user_data, user_token)
        
        # Required features (matching the order used in training)
        required_features = [
            "Sleep_Duration", "Calendar_Busy_Hours", "HeartRate_Avg", 
            "Steps_Last_24h", "Urgent_Emails_Flag"
        ]

        # Create the DataFrame required by the scikit-learn model
        features_df = pd.DataFrame([model_input_data], columns=required_features)

        # 3. MAKE PREDICTION
        # Prefer the model's native predict; otherwise adapt to the notebook engine
        stress_level = None
        try:
            if hasattr(DECISION_ENGINE, 'predict') and callable(getattr(DECISION_ENGINE, 'predict')):
                # Use predict if available
                prediction = DECISION_ENGINE.predict(features_df)[0]
                stress_level = int(prediction)
            elif hasattr(DECISION_ENGINE, 'stress_model') and getattr(DECISION_ENGINE, 'stress_model') is not None:
                # Prefer the model's own expected feature names if available
                sm = DECISION_ENGINE.stress_model
                if hasattr(sm, 'feature_names_in_'):
                    names = [str(n) for n in getattr(sm, 'feature_names_in_')]
                    # Build a flexible mapping from our current pipeline plus reasonable defaults
                    base_map = {
                        'Sleep_Duration': float(model_input_data.get('Sleep_Duration', 7.0)),
                        'Calendar_Busy_Hours': float(model_input_data.get('Calendar_Busy_Hours', 0.0)),
                        'Heart_Rate': float(model_input_data.get('HeartRate_Avg', 70.0)),
                        'Daily_Steps': float(model_input_data.get('Steps_Last_24h', 3000.0)),
                        'Urgent_Emails_Flag': float(model_input_data.get('Urgent_Emails_Flag', 0.0)),
                        # Defaults for likely notebook columns
                        'BMI_Category': 2.0,
                        'Systolic_BP': 120.0,
                        'Age': 35.0,
                        'Gender': 0.0,
                        'Occupation': 0.0,
                    }
                    vector = [float(base_map.get(name, 0.0)) for name in names]
                    prediction = sm.predict([vector])[0]
                    stress_level = int(prediction)
                elif hasattr(DECISION_ENGINE, 'feature_order'):
                    # Fallback to the engine's feature order if model doesn't expose names
                    feature_order = getattr(DECISION_ENGINE, 'feature_order')
                    mapped = {
                        'Sleep_Duration': float(model_input_data.get('Sleep_Duration', 7.0)),
                        'BMI_Category': 2.0,
                        'Heart_Rate': float(model_input_data.get('HeartRate_Avg', 70.0)),
                        'Daily_Steps': float(model_input_data.get('Steps_Last_24h', 3000.0)),
                        'Systolic_BP': 120.0
                    }
                    vector = [float(mapped.get(k, 0.0)) for k in feature_order]
                    prediction = sm.predict([vector])[0]
                    stress_level = int(prediction)
                else:
                    raise RuntimeError('stress_model lacks feature metadata')
            elif callable(DECISION_ENGINE):
                # Last resort: call engine(email_text, stress_features)
                emails = fetch_emails_for_urgency_analysis(user_token)
                email_text = emails[0] if emails else "SUBJECT: (none) BODY: (none)"
                stress_features = {
                    'Sleep_Duration': float(model_input_data.get('Sleep_Duration', 7.0)),
                    'BMI_Category': 2.0,
                    'Heart_Rate': float(model_input_data.get('HeartRate_Avg', 70.0)),
                    'Daily_Steps': float(model_input_data.get('Steps_Last_24h', 3000.0)),
                    'Systolic_BP': 120.0
                }
                result_obj = DECISION_ENGINE(email_text, stress_features)
                # Expect a dict with 'stress_level'
                if not isinstance(result_obj, dict) or 'stress_level' not in result_obj:
                    raise RuntimeError('Engine returned invalid result payload')
                stress_level = int(result_obj['stress_level'])
            else:
                raise RuntimeError('Loaded engine has no usable predict interface')
        except Exception as pred_err:
            raise HTTPException(status_code=503, detail=f"ML prediction failed: {pred_err}")
        
        # 4. MAP PREDICTION TO ACTION AND EXECUTE
        # Handle stress level prediction (0-10 scale from your trained model)
        
        # Get action payload based on stress level
        action_payload = ACTION_MAPPING.get(stress_level)
        
        if not action_payload:
            # Default action for unexpected predictions
            if stress_level <= 3:
                action_payload = ACTION_MAPPING[0]  # Low stress default
            elif stress_level <= 6:
                action_payload = ACTION_MAPPING[5]  # Medium stress default  
            else:
                action_payload = ACTION_MAPPING[7]  # High stress default

        # Create a copy to avoid modifying the original mapping
        action_request = action_payload.copy()
        action_request["user_token"] = user_token

        # Dispatch the action to the Actions Service
        action_response = requests.post(
            f"{ACTIONS_SERVICE_URL}/api/v1/execute_action",
            json=action_request,
            headers=headers,
            timeout=30
        )
        action_response.raise_for_status()
        
        return RecommendationResponse(
            status="success",
            recommendation=f"Action dispatched: {action_request['action']}",
            stress_level=stress_level,
            action_taken=action_request['action'],
            action_details=ActionDetails(**action_request['details']),
            features_used=FeatureData(**model_input_data),
            action_service_response=action_response.json(),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error communicating with upstream service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Comprehensive health check for the Assistant Service and its dependencies."
)
async def health_check():
    """
    Comprehensive health check endpoint that verifies:
    1. Assistant Service is running
    2. ML model is loaded
    3. Dependencies (Integrations & Actions services) are reachable
    4. NLP capabilities are available
    """
    health_status = {
        "status": "healthy",
        "service": "assistant",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {}
    }
    
    overall_healthy = True
    
    # Check 1: ML Model Status
    if DECISION_ENGINE:
        health_status["checks"]["ml_model"] = {
            "status": "healthy",
            "message": "Decision engine loaded successfully"
        }
    else:
        health_status["checks"]["ml_model"] = {
            "status": "unhealthy", 
            "message": "Decision engine not loaded"
        }
        overall_healthy = False
    
    # Check 2: NLP Capabilities
    if TRANSFORMERS_AVAILABLE:
        nlp_model_path = os.path.join(SCRIPT_DIR, "ml_models", "nlp_email_model")
        if os.path.exists(nlp_model_path):
            health_status["checks"]["nlp_model"] = {
                "status": "healthy",
                "message": "NLP model available and transformers loaded"
            }
        else:
            health_status["checks"]["nlp_model"] = {
                "status": "degraded",
                "message": "Transformers available but NLP model not found - using keyword fallback"
            }
    else:
        health_status["checks"]["nlp_model"] = {
            "status": "degraded",
            "message": "Transformers not available - using keyword-based urgency detection"
        }
    
    # Check 3: Integrations Service Connectivity
    try:
        response = requests.get(f"{INTEGRATIONS_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_status["checks"]["integrations_service"] = {
                "status": "healthy",
                "message": "Integrations service is reachable",
                "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2)
            }
        else:
            health_status["checks"]["integrations_service"] = {
                "status": "unhealthy",
                "message": f"Integrations service returned status {response.status_code}"
            }
            overall_healthy = False
    except requests.exceptions.RequestException as e:
        health_status["checks"]["integrations_service"] = {
            "status": "unhealthy",
            "message": f"Cannot reach integrations service: {str(e)}"
        }
        overall_healthy = False
    
    # Check 4: Actions Service Connectivity  
    try:
        response = requests.get(f"{ACTIONS_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_status["checks"]["actions_service"] = {
                "status": "healthy", 
                "message": "Actions service is reachable",
                "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2)
            }
        else:
            health_status["checks"]["actions_service"] = {
                "status": "unhealthy",
                "message": f"Actions service returned status {response.status_code}"
            }
            overall_healthy = False
    except requests.exceptions.RequestException as e:
        health_status["checks"]["actions_service"] = {
            "status": "unhealthy",
            "message": f"Cannot reach actions service: {str(e)}"
        }
        overall_healthy = False
    
    # Set overall status
    if not overall_healthy:
        health_status["status"] = "unhealthy"
    elif any(check["status"] == "degraded" for check in health_status["checks"].values()):
        health_status["status"] = "degraded"
    
    # Return appropriate HTTP status code
    if health_status["status"] == "unhealthy":
        raise HTTPException(
            status_code=503, 
            detail=ErrorResponse(
                detail="Service is unhealthy",
                error_code="SERVICE_UNHEALTHY",
                timestamp=datetime.now(timezone.utc).isoformat()
            ).dict()
        )
    else:
        return HealthResponse(**health_status)