import os
import datetime
import base64
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import requests
from dotenv import load_dotenv
from slack_sdk.webhook import WebhookClient

# Load environment variables from the .env file
load_dotenv()

app = FastAPI(title="Integrations Service")

# Scopes for Google APIs
SCOPES = [
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/calendar',  # For creating events
    'https://www.googleapis.com/auth/fitness.activity.read',
    'https://www.googleapis.com/auth/gmail.compose'  # For drafting emails
]

# A simple in-memory store for user credentials.
# In a real application, this would be a database.
credentials_store = {}

# Pydantic models for request validation
class CalendarEventRequest(BaseModel):
    title: str
    duration_minutes: int = 15
    description: str = ""

class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str

class SlackNotificationRequest(BaseModel):
    message: str
    channel: str = "#team-harmonia"

def get_credentials_from_token(authorization: Optional[str] = Header(None)):
    """
    Dependency to get credentials from JWT token or fallback to stored credentials.
    In a real app, this would validate the JWT and fetch user credentials from DB.
    """
    if authorization and authorization.startswith("Bearer "):
        # For now, we'll use the stored credentials regardless of the token
        # In production, you'd validate the JWT and fetch user-specific credentials
        token = authorization.split(" ")[1]
        # TODO: Validate JWT token and get user_id
        user_id = "user_id"  # This would come from the JWT
    else:
        user_id = "user_id"  # Fallback to default user
    
    creds = credentials_store.get(user_id)
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return creds

def get_credentials():
    """
    Legacy dependency for direct authentication (OAuth flow).
    """
    creds = credentials_store.get('user_id')
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return creds

@app.get("/auth/google")
async def google_auth():
    """
    Initiates the Google OAuth 2.0 authentication flow.
    """
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": os.getenv("CLIENT_ID"),
                "client_secret": os.getenv("CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [
                    "http://localhost:8001/auth/google/callback"
                ],
                "javascript_origins": [
                    "http://localhost:3000"
                ]
            }
        },
        scopes=SCOPES,
        redirect_uri='http://localhost:8001/auth/google/callback'
    )
    
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    
    # Store state to prevent CSRF attacks
    credentials_store['state'] = state
    
    return {"authorization_url": authorization_url}

@app.get("/auth/google/callback")
async def google_auth_callback(code: str, state: str):
    """
    Receives the authorization code from Google and exchanges it for a token.
    """
    if state != credentials_store.get('state'):
        raise HTTPException(status_code=400, detail="State mismatch.")
    
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": os.getenv("CLIENT_ID"),
                "client_secret": os.getenv("CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [
                    "http://localhost:8001/auth/google/callback"
                ],
                "javascript_origins": [
                    "http://localhost:3000"
                ]
            }
        },
        scopes=SCOPES,
        redirect_uri='http://localhost:8001/auth/google/callback'
    )
    
    flow.fetch_token(code=code)
    
    # Store credentials for the user (in this case, a hardcoded key)
    credentials_store['user_id'] = flow.credentials
    
    return {"message": "Authentication successful! You can now use the API."}

@app.get("/api/v1/data/calendar")
async def get_calendar_events(creds: Credentials = Depends(get_credentials)):
    """
    Fetches the user's calendar events for the next 24 hours.
    """
    try:
        service = build('calendar', 'v3', credentials=creds)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        return {"events": events_result.get('items', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/heart_rate")
async def get_heart_rate(creds: Credentials = Depends(get_credentials)):
    """
    Fetches heart rate data for the last 24 hours.
    """
    try:
        service = build('fitness', 'v1', credentials=creds)
        end_time_micros = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000000)
        start_time_micros = end_time_micros - (24 * 60 * 60 * 1000000)
        
        data_source = "derived:com.google.heart_rate.bpm:com.google.android.apps.fitness:blood_pressure"
        
        data_points = service.users().dataSources().datasets().get(
            userId="me",
            dataSourceId=data_source,
            datasetId=f"{start_time_micros}-{end_time_micros}"
        ).execute()
        
        return {"heart_rate_data": data_points.get("point", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/aggregate")
async def get_all_user_data(creds: Credentials = Depends(get_credentials)):
    """
    A single endpoint to get all data required by the Assistant Service.
    """
    calendar_events = await get_calendar_events(creds=creds)
    heart_rate_data = await get_heart_rate(creds=creds)
    
    # You can add more data points here
    
    return {
        "calendar_events": calendar_events["events"],
        "heart_rate_data": heart_rate_data["heart_rate_data"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.post("/api/v1/create_calendar_event")
async def create_calendar_event(
    event_request: CalendarEventRequest, 
    creds: Credentials = Depends(get_credentials_from_token)
):
    """
    Creates a calendar event for the user.
    """
    try:
        service = build('calendar', 'v3', credentials=creds)
        
        # Calculate start and end times
        start_time = datetime.datetime.now(datetime.timezone.utc)
        end_time = start_time + datetime.timedelta(minutes=event_request.duration_minutes)
        
        event = {
            'summary': event_request.title,
            'description': event_request.description,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'UTC',
            },
        }
        
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        
        return {
            "status": "success",
            "event_id": created_event.get('id'),
            "event_link": created_event.get('htmlLink'),
            "message": f"Event '{event_request.title}' created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create calendar event: {str(e)}")

@app.post("/api/v1/draft_email")
async def draft_email(
    email_request: EmailRequest, 
    creds: Credentials = Depends(get_credentials_from_token)
):
    """
    Creates a draft email using Gmail API.
    """
    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Create the email message
        message = f"To: {email_request.to}\nSubject: {email_request.subject}\n\n{email_request.body}"
        
        # Encode the message
        encoded_message = base64.urlsafe_b64encode(message.encode()).decode()
        
        draft = {
            'message': {
                'raw': encoded_message
            }
        }
        
        created_draft = service.users().drafts().create(userId='me', body=draft).execute()
        
        return {
            "status": "success",
            "draft_id": created_draft.get('id'),
            "message": f"Draft email to {email_request.to} created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create email draft: {str(e)}")

@app.post("/api/v1/send_slack_notification")
async def send_slack_notification(
    notification_request: SlackNotificationRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Sends a Slack notification using webhook.
    """
    try:
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        if not slack_webhook_url:
            return {
                "status": "mock_success",
                "message": f"Mock: Would send '{notification_request.message}' to {notification_request.channel}",
                "note": "Set SLACK_WEBHOOK_URL environment variable for actual Slack integration"
            }
        
        # Use Slack SDK for better handling
        webhook = WebhookClient(slack_webhook_url)
        response = webhook.send(
            text=notification_request.message
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": f"Notification sent to {notification_request.channel}"
            }
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to send Slack notification")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send Slack notification: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for service monitoring.
    """
    return {"status": "healthy", "service": "integrations"}