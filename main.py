# main.py - Gateway server for all chatbots
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import all bots
from bots.salt_bot import SaltBot
from bots.ecommerce_bot import EcommerceBot
# from bots.ecommerce_bot import EcommerceBot  # Add others similarly
# from bots.realestate_bot import RealEstateBot
# from bots.smoking_bot import SmokingBot

app = FastAPI(title="Multi-Chatbot Gateway", description="Single server for multiple website chatbots")

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all bots
bots = {
    "salt": SaltBot(),
    "ecommerce": EcommerceBot(),
    # "realestate": RealEstateBot(),
    # "smoking": SmokingBot(),
}

print(f"\n✅ Loaded {len(bots)} bots: {', '.join(bots.keys())}\n")

# --- REQUEST/RESPONSE MODELS ---
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    status: str
    bot_name: str
    bot_id: str
    context_used: Optional[List[str]] = None
    timestamp: str

# --- API ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "message": "Multi-Chatbot Gateway API",
        "status": "online",
        "available_bots": list(bots.keys()),
        "bots": {bot_id: bot.name for bot_id, bot in bots.items()}
    }

@app.post("/api/chat/{bot_id}", response_model=ChatResponse)
async def chat_with_bot(bot_id: str, request: ChatRequest):
    """Chat with a specific bot: /api/chat/salt, /api/chat/ecommerce"""
    
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    
    bot = bots[bot_id]
    result = await bot.chat(
        message=request.message,
        user_id=request.user_id,
        conversation_history=request.conversation_history
    )
    
    return ChatResponse(**result)

@app.post("/api/chat")
async def chat_auto(request: ChatRequest, website: Optional[str] = None):
    """Auto-detect bot from website parameter or origin header"""
    
    bot_id = website
    
    # If not specified, try to detect from origin header
    if not bot_id:
        origin = request.headers.get("origin", "")
        if "salt" in origin or "saltweb" in origin:
            bot_id = "salt"
        # Add other domain mappings
        # elif "ecom" in origin:
        #     bot_id = "ecommerce"
    
    if not bot_id or bot_id not in bots:
        raise HTTPException(status_code=400, detail="Could not determine which bot to use")
    
    bot = bots[bot_id]
    result = await bot.chat(
        message=request.message,
        user_id=request.user_id,
        conversation_history=request.conversation_history
    )
    
    return result

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    bot_status = {}
    for bot_id, bot in bots.items():
        bot_status[bot_id] = {
            "name": bot.name,
            "status": "healthy",
            "groq": "connected" if hasattr(bot, 'groq_client') and bot.groq_client else "unknown",
            "supabase": "connected" if hasattr(bot, 'supabase') and bot.supabase else "unknown"
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bots": bot_status,
        "total_bots": len(bots)
    }

@app.get("/api/bots")
async def list_bots():
    """List all available bots"""
    return {
        "bots": [
            {
                "id": bot_id,
                "name": bot.name,
                "description": getattr(bot, 'description', 'No description')
            }
            for bot_id, bot in bots.items()
        ]
    }

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🤖 MULTI-BOT GATEWAY SERVER")
    print("="*60)
    print(f"📍 Server: http://0.0.0.0:8000")
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"🚀 Available bots: {', '.join(bots.keys())}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)