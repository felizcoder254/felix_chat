from fastapi import FastAPI, HTTPException
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import requests

app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model Setup ===
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# === Load FAQ Data ===
faq_path = "documents/zendawa_faq.txt"
with open(faq_path, "r", encoding="utf-8") as f:
    raw_faq_blocks = f.read().strip().split("\n\n")

faq_pairs = []
for block in raw_faq_blocks:
    lines = block.strip().split("\n")
    q = next((line[3:] for line in lines if line.lower().startswith("q:")), None)
    a = next((line[3:] for line in lines if line.lower().startswith("a:")), None)
    if q and a:
        faq_pairs.append((q.strip(), a.strip()))

questions = [q for q, _ in faq_pairs]
answers = [a for _, a in faq_pairs]
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

@app.post("/chat")
async def chat(chat_req: ChatRequest):
    if not TOGETHER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Together.ai API key")

    user_msg = chat_req.messages[-1].content.strip()
    low = user_msg.lower()

    # Handle drug ordering intent
    if any(kw in low for kw in ["order drug", "buy drug", "order meds", "purchase drug"]):
        return {"reply": (
            "🛒 To order drugs on NCK Helper:\n"
            "1. Go to our Marketplace: https://nckhelper.example.com/marketplace\n"
            "2. Browse or search for the medications you need.\n"
            "3. Add your selections to the cart.\n"
            "4. Proceed to checkout and enter delivery details.\n"
            "5. Complete payment and confirm your order.\n\n"
            "👉 Start ordering here: https://nckhelper.example.com/marketplace"
        )}

    # Handle consultation intent
    if any(kw in low for kw in ["consultation", "consult", "teleconsult", "book a doctor"]):
        return {"reply": (
            "💬 To book a consultation on NCK Helper:\n"
            "1. Visit: https://nckhelper.example.com/consultations\n"
            "2. Sign in or create a NCK Helper account.\n"
            "3. Select your desired service and provider.\n"
            "4. Choose a date, time, and mode (chat/video).\n"
            "5. Confirm and await your appointment link.\n\n"
            "👉 Book now: https://nckhelper.example.com/consultations"
        )}

    # Handle known staff queries
    name_responses = {
        "achoka": "Dr. Victor Achoka is a co-founder of NCK Helper and a licensed pharmacist.",
        "chege": "Wilfred Chege is the CEO and co-founder of NCK Helper, leading our mission.",
        "vincent chebon": "Vincent Chebon is the CTO at NCK Helper, driving technical innovation."
    }
    for name, resp in name_responses.items():
        if name in low:
            return {"reply": resp}

    # FAQ similarity + LLM fallback
    user_vector = vectorizer.transform([user_msg])
    sims = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = int(sims.argmax())

    if sims[best_idx] < 0.3:
        return {"reply": (
            "I'm here to help with NCK Helper’s medical exam assistant services. "
            "Please ask about consultations, drug ordering, exam assistance, or healthcare support."
        )}

    matched_q = questions[best_idx]
    matched_a = answers[best_idx]

    system_prompt = ("You are NCK Helper Assistant, a helpful AI designed to support users with accurate and friendly information "
                     "about NCK Helper — a medical exam assistant platform offering services like drug ordering, pharmacy onboarding, "
                     "teleconsultations, and healthcare logistics.\n\n"
                     "If a question falls outside NCK Helper’s scope, kindly guide the user to ask relevant questions.\n\n"
                     f"FAQ match:\nQ: {matched_q}\nA: {matched_a}")

    prompt_messages = [{"role": "system", "content": system_prompt}] + [msg.dict() for msg in chat_req.messages]
    payload = {"model": MODEL, "messages": prompt_messages}
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, I don't have that info.")
        return {"reply": reply}
    except Exception:
        return {"reply": "Sorry, I could not retrieve a response. Please try again later."}

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return Path("static/index.html").read_text()
