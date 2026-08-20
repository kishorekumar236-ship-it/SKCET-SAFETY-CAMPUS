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

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
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
    
    print(f"[BACKEND] Received: {request.message}")
    
    conversation_history.append({"role": "user", "content": request.message})
    
    try:
        prompt = SYSTEM_PROMPT + "\n\n"
        for msg in conversation_history[-6:]:
            prompt += f"{msg['role'].title()}: {msg['content']}\n"
        prompt += "Assistant: "
        
        print(f"[BACKEND] Calling Ollama at {OLLAMA_URL}")
        
        response = requests.post(
            OLLAMA_URL,
            json={"model": "llama2", "prompt": prompt, "stream": False},
            timeout=120
        )
        
        if response.status_code != 200:
            print(f"[BACKEND] Ollama error: {response.status_code}")
            raise Exception(f"Ollama error: {response.status_code}")
        
        data = response.json()
        reply = data.get("response", "").strip()
        
        print(f"[BACKEND] Got reply: {reply[:50]}...")
        
        conversation_history.append({"role": "assistant", "content": reply})
        
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        return {"reply": reply}
    
    except Exception as e:
        error_msg = str(e)
        print(f"[BACKEND] ERROR: {error_msg}")
        return {"reply": f"Backend error: {error_msg}", "error": error_msg}