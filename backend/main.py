import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_history = []

SYSTEM_PROMPT = """You are SKCET Campus Safety AI Assistant.
Answer VERY BRIEFLY (1-2 sentences only).
Help with: campus safety, emergencies (112), incident reporting, security.
Keep it SHORT."""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "ok", "message": "SKCET Chatbot API"}

# FIXED: Use localhost, not docker hostname
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

@app.post("/chat")
def chat(request: ChatRequest):
    global conversation_history
    
    conversation_history.append({"role": "user", "content": request.message})
    
    try:
        prompt = SYSTEM_PROMPT + "\n\n"
        for msg in conversation_history[-6:]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += "Assistant: "
        
        # FIXED: Added max_tokens=50 for faster responses
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False,
                "num_predict": 25  # Limit response length
            },
            timeout=60
        )
        
        reply = response.json().get("response", "").strip()
        conversation_history.append({"role": "assistant", "content": reply})
        
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        return {"reply": reply}
    
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "error": str(e)}
