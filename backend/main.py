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

SYSTEM_PROMPT = """You are SKCET Campus Safety Chatbot. Help with:
- Emergency procedures (call 112)
- Incident reporting
- Campus safety advice
- Emergency exits and evacuation
- Security contacts: safety@skcet.ac.in
- Fire safety
Keep responses short and practical."""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "ok", "message": "SKCET Chatbot API"}

@app.post("/chat")
def chat(request: ChatRequest):
    global conversation_history
    
    conversation_history.append({
        "role": "user",
        "content": request.message
    })
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + conversation_history
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        reply = response.choices[0].message.content
        
        conversation_history.append({
            "role": "assistant",
            "content": reply
        })
        
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        return {"reply": reply}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"reply": f"Error: {str(e)}", "error": str(e)}