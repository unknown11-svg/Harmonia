import os
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI(title="Actions Service")

# URL for the Integrations Service
# Use localhost for local development, Docker service name for containerized deployment
INTEGRATIONS_SERVICE_URL = os.getenv("INTEGRATIONS_SERVICE_URL", "http://localhost:8001")

class ActionRequest(BaseModel):
    action: str
    user_token: Optional[str] = None
    details: Dict[str, Any] = {}

@app.post("/api/v1/execute_action")
async def execute_action(action_request: ActionRequest):
    """
    Receives an action command from the Assistant Service and executes it.
    """
    try:
        action_name = action_request.action
        user_token = action_request.user_token
        details = action_request.details

        if not action_name:
            raise HTTPException(status_code=400, detail="Action name is required.")

        headers = {}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"
        
        # Dispatch the action based on the payload
        if action_name == "create_break_event":
            # The Actions service is just a middleman, it tells the Integrations service what to do.
            response = requests.post(
                f"{INTEGRATIONS_SERVICE_URL}/api/v1/create_calendar_event",
                json={
                    "title": details.get("title", "Quick Break"), 
                    "duration_minutes": details.get("duration", 15),
                    "description": details.get("description", "AI-suggested break time")
                },
                headers=headers,
                timeout=30
            )
        
        elif action_name == "draft_email":
            response = requests.post(
                f"{INTEGRATIONS_SERVICE_URL}/api/v1/draft_email",
                json={
                    "to": details.get("to"), 
                    "subject": details.get("subject"), 
                    "body": details.get("body")
                },
                headers=headers,
                timeout=30
            )

        elif action_name == "send_slack_notification":
            response = requests.post(
                f"{INTEGRATIONS_SERVICE_URL}/api/v1/send_slack_notification",
                json={
                    "message": details.get("message"),
                    "channel": details.get("channel", "#team-harmonia")
                },
                headers=headers,
                timeout=30
            )
               
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action_name}")

        response.raise_for_status()
        
        # Return both the action result and the response from integrations service
        integration_response = response.json()
        return {
            "status": "success", 
            "action": action_name, 
            "message": "Action executed successfully.",
            "integration_response": integration_response
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout while communicating with Integrations Service")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to Integrations Service")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error communicating with Integrations Service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for service monitoring.
    """
    try:
        # Test connection to integrations service
        response = requests.get(f"{INTEGRATIONS_SERVICE_URL}/health", timeout=5)
        integrations_healthy = response.status_code == 200
    except:
        integrations_healthy = False
        
    return {
        "status": "healthy" if integrations_healthy else "degraded",
        "service": "actions",
        "dependencies": {
            "integrations_service": "healthy" if integrations_healthy else "unhealthy"
        }
    }

@app.get("/")
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": "Actions Service",
        "version": "1.0.0",
        "description": "Executes actions by coordinating with the Integrations Service",
        "endpoints": {
            "execute_action": "/api/v1/execute_action",
            "health": "/health"
        }
    }