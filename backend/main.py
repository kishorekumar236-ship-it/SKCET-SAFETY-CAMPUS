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

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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

import requests

OLLAMA_URL = "http://ollama:11434/api/generate"

@app.post("/chat")
def chat(request: ChatRequest):
    global conversation_history
    
    conversation_history.append({"role": "user", "content": request.message})
    
    try:
        prompt = SYSTEM_PROMPT + "\n\n"
        for msg in conversation_history[-6:]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += "Assistant: "
        
        response = requests.post(
            OLLAMA_URL,
            json={"model": "llama2", "prompt": prompt, "stream": False},
            timeout=60
        )
        
        reply = response.json().get("response", "").strip()
        conversation_history.append({"role": "assistant", "content": reply})
        
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        return {"reply": reply}
    
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "error": str(e)}
