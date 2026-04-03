from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import bcrypt

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Supabase connection
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

JWT_SECRET = os.environ.get('JWT_SECRET', 'default_secret')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7

security = HTTPBearer()

app = FastAPI()
api_router = APIRouter(prefix="/api")

# ============ MODELS ============

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: str

class TokenResponse(BaseModel):
    token: str
    user: UserResponse

class ProgressUpdate(BaseModel):
    business_id: str
    step_id: str
    completed: bool

class AIAssistantRequest(BaseModel):
    business_id: str
    problem_context: str
    current_step: Optional[str] = None
    language: Optional[str] = "en"

class AIRecommendationRequest(BaseModel):
    business_type: str
    budget: Optional[str] = "moderate"
    experience: Optional[str] = "beginner"
    language: Optional[str] = "en"

# Admin email for restricted access
ADMIN_EMAIL = "elianmh21@gmail.com"

# ============ ADMIN MODELS ============

class AdminNoteCreate(BaseModel):
    title: str
    content: Optional[str] = ""
    category: Optional[str] = "general"
    is_checklist: Optional[bool] = False
    checklist_items: Optional[List[dict]] = []
    priority: Optional[str] = "medium"
    is_pinned: Optional[bool] = False

class AdminNoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    is_checklist: Optional[bool] = None
    checklist_items: Optional[List[dict]] = None
    priority: Optional[str] = None
    is_pinned: Optional[bool] = None

class ChecklistItemUpdate(BaseModel):
    note_id: int
    item_index: int
    completed: bool

# ============ DETAILED BUSINESS GUIDES ============

BUSINESS_GUIDES = [
    {
        "id": "cleaning-business",
        "title": "Cleaning Company (Residential & Commercial)",
        "title_es": "Compañía de Limpieza (Residencial y Comercial)",
        "description": "Start a cleaning business serving homes, offices, or restaurants. High demand, recurring revenue, and perfect for lead generation.",
        "description_es": "Inicia un negocio de limpieza sirviendo casas, oficinas o restaurantes. Alta demanda, ingresos recurrentes y perfecto para generación de leads.",
        "scale_strategy": "Start solo with residential clients. Add commercial contracts. Hire cleaners and become the manager.",
        "scale_strategy_es": "Comienza solo con clientes residenciales. Añade contratos comerciales. Contrata limpiadores y conviértete en gerente.",
        "icon": "sparkles",
        "market_analysis": {
            "market_size": "$74B+ cleaning services market",
            "growth_rate": "6% annual growth",
            "startup_cost": "$200 - $2,000",
            "time_to_revenue": "3-7 days",
            "profit_margin": "40-70%",
            "difficulty": "Easy"
        },
        "market_analysis_es": {
            "market_size": "Mercado de limpieza $74B+",
            "growth_rate": "6% crecimiento anual",
            "startup_cost": "$200 - $2,000",
            "time_to_revenue": "3-7 días",
            "profit_margin": "40-70%",
            "difficulty": "Fácil"
        },
        "common_problems": [
            {"id": "cp1", "problem": "Client cancels last minute", "problem_es": "Cliente cancela a último minuto", "solution": "Implement 24-48hr cancellation policy. Charge 50% for late cancellations. Keep a waitlist of backup clients.", "solution_es": "Implementa política de cancelación 24-48hrs. Cobra 50% por cancelaciones tardías. Mantén lista de espera de clientes de respaldo."},
            {"id": "cp2", "problem": "Client claims something is broken/missing", "problem_es": "Cliente reclama que algo está roto/faltante", "solution": "Always take before/after photos. Have liability insurance. Document everything in writing.", "solution_es": "Siempre toma fotos antes/después. Ten seguro de responsabilidad. Documenta todo por escrito."},
            {"id": "cp3", "problem": "Client doesn't pay or delays payment", "problem_es": "Cliente no paga o retrasa el pago", "solution": "Require payment before or immediately after service. Use digital invoicing with automatic reminders.", "solution_es": "Requiere pago antes o inmediatamente después del servicio. Usa facturación digital con recordatorios automáticos."},
            {"id": "cp4", "problem": "Not enough clients initially", "problem_es": "No hay suficientes clientes inicialmente", "solution": "Offer 50% off first cleaning for reviews. Post daily on Nextdoor, Facebook groups. Ask every friend/family for referrals.", "solution_es": "Ofrece 50% de descuento en primera limpieza por reseñas. Publica diario en Nextdoor, grupos de Facebook. Pide referidos a amigos/familia."},
            {"id": "cp5", "problem": "Physical exhaustion / burnout", "problem_es": "Agotamiento físico / burnout", "solution": "Limit to 3-4 houses per day max. Raise prices to work less. Hire helper when you hit 15+ recurring clients.", "solution_es": "Limita a 3-4 casas por día máximo. Sube precios para trabajar menos. Contrata ayudante cuando tengas 15+ clientes recurrentes."}
        ],
        "steps": []
    }
]

# ============ 99 FLORIDA BUSINESS OPPORTUNITIES ============

BUSINESS_CATEGORIES = [
    {"id": "lawn-landscaping", "num": 1, "title": "Lawn & Landscaping Básico", "title_es": "Jardinería y Paisajismo Básico", "icon": "tree", "category": "servicios", "capital": "$1.5k–3k", "tiempo": "2–4 sem"},
    {"id": "pressure-washing", "num": 2, "title": "Pressure Washing", "title_es": "Lavado a Presión", "icon": "droplets", "category": "servicios", "capital": "$1.5k–3k", "tiempo": "1–3 sem"},
    {"id": "window-cleaning", "num": 3, "title": "Window Cleaning Residencial", "title_es": "Limpieza de Ventanas Residencial", "icon": "sparkles", "category": "limpieza", "capital": "$1.5k–3k", "tiempo": "1–2 sem"},
    {"id": "house-cleaning", "num": 4, "title": "House Cleaning / Maid Service", "title_es": "Limpieza de Casas", "icon": "sparkles", "category": "limpieza", "capital": "$1.5k–3k", "tiempo": "1–2 sem"},
    {"id": "handyman", "num": 5, "title": "Handyman sin Contrato General", "title_es": "Handyman sin Contrato General", "icon": "wrench", "category": "servicios", "capital": "$1.5k–3k", "tiempo": "2–3 sem"},
]

# ============ AUTH HELPERS ============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        result = supabase.table("users").select("id, email, name, created_at").eq("id", user_id).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=401, detail="User not found")
        return result.data[0]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============ AUTH ROUTES ============

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(data: UserCreate):
    existing = supabase.table("users").select("id").eq("email", data.email).execute()
    if existing.data and len(existing.data) > 0:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": data.email,
        "name": data.name,
        "password": hash_password(data.password),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    supabase.table("users").insert(user_doc).execute()
    token = create_token(user_id)
    return TokenResponse(token=token, user=UserResponse(id=user_id, email=data.email, name=data.name, created_at=user_doc["created_at"]))

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(data: UserLogin):
    result = supabase.table("users").select("*").eq("email", data.email).execute()
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user = result.data[0]
    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["id"])
    return TokenResponse(token=token, user=UserResponse(id=user["id"], email=user["email"], name=user["name"], created_at=user["created_at"]))

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user)):
    return UserResponse(id=user["id"], email=user["email"], name=user["name"], created_at=user["created_at"])

# ============ STATS ============

@api_router.get("/stats/members")
async def get_member_count():
    result = supabase.table("users").select("id", count="exact").execute()
    count = result.count if result.count else 0
    return {"member_count": count, "updated_at": datetime.now(timezone.utc).isoformat()}

# ============ BUSINESS ROUTES ============

@api_router.get("/businesses")
async def get_businesses():
    return BUSINESS_GUIDES

@api_router.get("/businesses/categories")
async def get_business_categories():
    return BUSINESS_CATEGORIES

@api_router.get("/businesses/{business_id}")
async def get_business(business_id: str):
    for business in BUSINESS_GUIDES:
        if business["id"] == business_id:
            return business
    raise HTTPException(status_code=404, detail="Business not found")

# ============ PROGRESS ============

@api_router.get("/progress")
async def get_user_progress(user: dict = Depends(get_current_user)):
    result = supabase.table("progress").select("*").eq("user_id", user["id"]).execute()
    return result.data if result.data else []

@api_router.get("/progress/{business_id}")
async def get_business_progress(business_id: str, user: dict = Depends(get_current_user)):
    result = supabase.table("progress").select("*").eq("user_id", user["id"]).eq("business_id", business_id).execute()
    if not result.data or len(result.data) == 0:
        return {"user_id": user["id"], "business_id": business_id, "completed_steps": [], "updated_at": None}
    return result.data[0]

@api_router.post("/progress")
async def update_progress(data: ProgressUpdate, user: dict = Depends(get_current_user)):
    existing = supabase.table("progress").select("*").eq("user_id", user["id"]).eq("business_id", data.business_id).execute()
    if existing.data and len(existing.data) > 0:
        completed_steps = existing.data[0].get("completed_steps", []) or []
        if data.completed and data.step_id not in completed_steps:
            completed_steps.append(data.step_id)
        elif not data.completed and data.step_id in completed_steps:
            completed_steps.remove(data.step_id)
        supabase.table("progress").update({
            "completed_steps": completed_steps,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("user_id", user["id"]).eq("business_id", data.business_id).execute()
    else:
        completed_steps = [data.step_id] if data.completed else []
        supabase.table("progress").insert({
            "user_id": user["id"],
            "business_id": data.business_id,
            "completed_steps": completed_steps,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    result = supabase.table("progress").select("*").eq("user_id", user["id"]).eq("business_id", data.business_id).execute()
    return result.data[0] if result.data else {"user_id": user["id"], "business_id": data.business_id, "completed_steps": completed_steps}

# ============ AI ASSISTANT ============

@api_router.post("/ai/assistant")
async def ai_assistant(data: AIAssistantRequest):
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="AI service not configured")
        business = next((b for b in BUSINESS_GUIDES if b["id"] == data.business_id), None)
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        lang = "Spanish" if data.language == "es" else "English"
        business_title = business["title_es"] if data.language == "es" else business["title"]
        problems_context = ""
        for prob in business.get("common_problems", []):
            p = prob["problem_es"] if data.language == "es" else prob["problem"]
            s = prob["solution_es"] if data.language == "es" else prob["solution"]
            problems_context += f"- Problem: {p}\n  Solution: {s}\n"
        system_message = f"""You are an expert business advisor for {business_title}. Respond in {lang}. Be practical, specific, and actionable.\n\nKnown problems and solutions:\n{problems_context}"""
        chat = LlmChat(api_key=api_key, session_id=f"assistant-{data.business_id}-{uuid.uuid4()}", system_message=system_message).with_model("anthropic", "claude-sonnet-4-5-20250929")
        context = f"Current step: {data.current_step}\n" if data.current_step else ""
        response = await chat.send_message(UserMessage(text=f"{context}User's question: {data.problem_context}"))
        return {"response": response, "business_id": data.business_id, "business_title": business_title}
    except ImportError:
        raise HTTPException(status_code=500, detail="AI integration not available")
    except Exception as e:
        logging.error(f"AI assistant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/ai/recommend")
async def get_ai_recommendation(data: AIRecommendationRequest):
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="AI service not configured")
        lang_instruction = "Respond in Spanish." if data.language == "es" else "Respond in English."
        chat = LlmChat(api_key=api_key, session_id=f"recommendation-{uuid.uuid4()}", system_message=f"You are an expert business consultant. {lang_instruction}").with_model("anthropic", "claude-sonnet-4-5-20250929")
        response = await chat.send_message(UserMessage(text=f"I want to start a {data.business_type} business. Budget: {data.budget}, Experience: {data.experience}. Give me 3 specific actions for THIS WEEK, investment breakdown, and timeline to first customer."))
        return {"recommendation": response, "business_type": data.business_type}
    except Exception as e:
        logging.error(f"AI error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ ADMIN HELPERS ============

async def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        result = supabase.table("users").select("*").eq("id", user_id).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=401, detail="User not found")
        user = result.data[0]
        if user["email"] != ADMIN_EMAIL:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============ ADMIN ROUTES ============

@api_router.get("/admin/check")
async def check_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            return {"is_admin": False}
        result = supabase.table("users").select("email").eq("id", user_id).execute()
        if not result.data or len(result.data) == 0:
            return {"is_admin": False}
        return {"is_admin": result.data[0]["email"] == ADMIN_EMAIL}
    except Exception:
        return {"is_admin": False}

@api_router.get("/admin/users")
async def get_all_users(admin: dict = Depends(verify_admin)):
    try:
        result = supabase.table("users").select("id, email, name, created_at").order("created_at", desc=True).execute()
        return {"users": result.data if result.data else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/stats")
async def get_admin_stats(admin: dict = Depends(verify_admin)):
    try:
        users_result = supabase.table("users").select("id", count="exact").execute()
        progress_result = supabase.table("progress").select("user_id", count="exact").execute()
        return {
            "total_users": users_result.count or 0,
            "active_users": progress_result.count or 0,
            "total_businesses": 99,
            "detailed_guides": 5
        }
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")
        return {"total_users": 0, "active_users": 0, "total_businesses": 99, "detailed_guides": 5}

@api_router.get("/admin/notes")
async def get_admin_notes(admin: dict = Depends(verify_admin)):
    try:
        result = supabase.table("admin_notes").select("*").eq("admin_id", admin["id"]).order("is_pinned", desc=True).order("created_at", desc=True).execute()
        return {"notes": result.data if result.data else []}
    except Exception as e:
        logging.error(f"Error fetching notes: {e}")
        return {"notes": []}

@api_router.post("/admin/notes")
async def create_admin_note(note: AdminNoteCreate, admin: dict = Depends(verify_admin)):
    try:
        import json
        note_data = {
            "admin_id": admin["id"],
            "title": note.title,
            "content": note.content,
            "category": note.category,
            "is_checklist": note.is_checklist,
            "checklist_items": json.dumps(note.checklist_items) if note.checklist_items else "[]",
            "priority": note.priority,
            "is_pinned": note.is_pinned,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        result = supabase.table("admin_notes").insert(note_data).execute()
        return {"note": result.data[0] if result.data else note_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/admin/notes/{note_id}")
async def update_admin_note(note_id: int, note: AdminNoteUpdate, admin: dict = Depends(verify_admin)):
    try:
        import json
        update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if note.title is not None: update_data["title"] = note.title
        if note.content is not None: update_data["content"] = note.content
        if note.category is not None: update_data["category"] = note.category
        if note.is_checklist is not None: update_data["is_checklist"] = note.is_checklist
        if note.checklist_items is not None: update_data["checklist_items"] = json.dumps(note.checklist_items)
        if note.priority is not None: update_data["priority"] = note.priority
        if note.is_pinned is not None: update_data["is_pinned"] = note.is_pinned
        result = supabase.table("admin_notes").update(update_data).eq("id", note_id).eq("admin_id", admin["id"]).execute()
        return {"note": result.data[0] if result.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/notes/{note_id}")
async def delete_admin_note(note_id: int, admin: dict = Depends(verify_admin)):
    try:
        supabase.table("admin_notes").delete().eq("id", note_id).eq("admin_id", admin["id"]).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/admin/notes/{note_id}/checklist")
async def update_checklist_item(note_id: int, update: ChecklistItemUpdate, admin: dict = Depends(verify_admin)):
    try:
        import json
        result = supabase.table("admin_notes").select("checklist_items").eq("id", note_id).eq("admin_id", admin["id"]).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Note not found")
        items = result.data[0].get("checklist_items", [])
        if isinstance(items, str):
            items = json.loads(items)
        if update.item_index < len(items):
            items[update.item_index]["completed"] = update.completed
        supabase.table("admin_notes").update({
            "checklist_items": json.dumps(items),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", note_id).eq("admin_id", admin["id"]).execute()
        return {"success": True, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ MAIN ============

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@app.on_event("shutdown")
async def shutdown_db_client():
    pass  # supabase client does not require explicit closing
