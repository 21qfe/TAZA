from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatbotMessageRequest(BaseModel):
    input: str = Field(..., description="User query or task command")
    role: str = Field("farmer", description="Active user role: 'farmer' or 'consumer'")
    mode: str = Field("chat", description="Active mode: 'chat' or 'task'")
    model: str = Field("TAZA-Qwen2.5-0.5B-LoRA (IBM Z LinuxONE s390x)", description="Active AI model: default TAZA LinuxONE")
    api_key: Optional[str] = Field(None, description="Optional custom API Key")
    district: Optional[str] = Field("Hooghly", description="Target agricultural district")
    farmer_id: Optional[int] = Field(1, description="Active farmer ID for database operations")


class AutomatedTaskResult(BaseModel):
    type: str
    title: str
    executed: bool = True
    deployed: bool = True
    timestamp: str
    details: Dict[str, Any] = Field(default_factory=dict)
    html_card: Optional[str] = None


class ChatbotMessageResponse(BaseModel):
    success: bool = True
    role: str
    mode: str
    model_used: str
    output_text: str
    automated_task: Optional[AutomatedTaskResult] = None
    api_status: str = "connected"
