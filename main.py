# main.py - Gateway server for all chatbots with multimodal support
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import base64
from io import BytesIO

# Import all bots
from bots.salt_bot import SaltBot
from bots.ecommerce_bot import EcommerceBot
from bots.realestate_bot import RealEstateBot
from bots.smoking import SmokingBot

app = FastAPI(
    title="Multi-Chatbot Gateway", 
    description="Single server for multiple website chatbots with multimodal support (text, voice, image, PDF)",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all bots
bots = {
    "salt": SaltBot(),
    "ecommerce": EcommerceBot(),
    "realestate": RealEstateBot(),
    "smoking": SmokingBot(),
}

print(f"\n Loaded {len(bots)} bots: {', '.join(bots.keys())}\n")

# PYDANTIC MODELS 
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
    transcribed_text: Optional[str] = None
    image_analysis: Optional[str] = None
    pdf_analysis: Optional[Dict] = None
    timestamp: str

class BotInfo(BaseModel):
    id: str
    name: str
    description: str
    capabilities: List[str]

# GENERIC MULTIMODAL HANDLERS

async def handle_multimodal_request(bot_id: str, request_type: str, **kwargs):
    """
    Generic handler for multimodal requests
    Any bot can implement these methods:
    - chat_with_voice()
    - chat_with_image()  
    - chat_with_pdf()
    """
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    
    bot = bots[bot_id]
    
    # Check if bot supports the requested multimodal feature
    if request_type == "voice" and not hasattr(bot, 'chat_with_voice'):
        raise HTTPException(status_code=501, detail=f"Bot '{bot_id}' does not support voice input")
    elif request_type == "image" and not hasattr(bot, 'chat_with_image'):
        raise HTTPException(status_code=501, detail=f"Bot '{bot_id}' does not support image input")
    elif request_type == "pdf" and not hasattr(bot, 'chat_with_pdf'):
        raise HTTPException(status_code=501, detail=f"Bot '{bot_id}' does not support PDF input")    
    if request_type == "voice":
        return await bot.chat_with_voice(**kwargs)
    elif request_type == "image":
        return await bot.chat_with_image(**kwargs)
    elif request_type == "pdf":
        return await bot.chat_with_pdf(**kwargs)

#  API ENDPOINTS 

@app.get("/")
async def root():
    return {
        "message": "Multi-Chatbot Gateway API with Multimodal Support",
        "status": "online",
        "version": "2.0.0",
        "available_bots": list(bots.keys()),
        "bots": {
            bot_id: {
                "name": bot.name,
                "capabilities": get_capabilities(bot)
            } 
            for bot_id, bot in bots.items()
        }
    }

def get_capabilities(bot) -> List[str]:
    """Detect bot capabilities"""
    caps = ["text"]
    if hasattr(bot, 'chat_with_voice'):
        caps.append("voice")
    if hasattr(bot, 'chat_with_image'):
        caps.append("image")
    if hasattr(bot, 'chat_with_pdf'):
        caps.append("pdf")
    return caps

# ============ TEXT CHAT ENDPOINTS ============

@app.post("/api/chat/{bot_id}", response_model=ChatResponse)
async def chat_with_bot(bot_id: str, request: ChatRequest):
    """Chat with a specific bot via text"""
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
    
    if not bot_id:
        origin = request.headers.get("origin", "")
        # Domain mapping
        domain_map = {
            "salt": ["salt", "saltweb"],
            "ecommerce": ["ecom", "shop", "store"],
            "realestate": ["realestate", "property", "housing"],
            "smoking": ["smoking", "vape", "cigarette"]
        }
        
        for bot, domains in domain_map.items():
            if any(domain in origin.lower() for domain in domains):
                bot_id = bot
                break
    
    if not bot_id or bot_id not in bots:
        raise HTTPException(status_code=400, detail="Could not determine which bot to use")
    
    bot = bots[bot_id]
    result = await bot.chat(
        message=request.message,
        user_id=request.user_id,
        conversation_history=request.conversation_history
    )
    
    return result

# VOICE ENDPOINTS 

@app.post("/api/chat/{bot_id}/voice")
async def chat_with_voice(
    bot_id: str,
    audio: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    language: str = Form("en")
):
    """
    Send voice message to bot
    Accepts audio files (wav, mp3, etc.)
    """
    try:
        audio_bytes = await audio.read()        
        result = await handle_multimodal_request(
            bot_id, 
            "voice",
            audio_data=audio_bytes,
            user_id=user_id,
            language=language
        )
        
        return {
            "response": result.get("response"),
            "transcribed_text": result.get("transcribed_text", ""),
            "status": result.get("status"),
            "bot_name": result.get("bot_name"),
            "bot_id": result.get("bot_id"),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Voice endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

# IMAGE ENDPOINTS
@app.post("/api/chat/{bot_id}/image")
async def chat_with_image(
    bot_id: str,
    image: UploadFile = File(...),
    message: Optional[str] = Form("What is this product?"),
    user_id: Optional[str] = Form(None)
):
    """
    Send image to bot for analysis
    Accepts image files (jpg, png, etc.)
    """
    try:
        image_bytes = await image.read()        
        result = await handle_multimodal_request(
            bot_id,
            "image",
            message=message,
            image_data=image_bytes,
            user_id=user_id
        )
        
        return {
            "response": result.get("response"),
            "image_analysis": result.get("image_analysis", ""),
            "status": result.get("status"),
            "bot_name": result.get("bot_name"),
            "bot_id": result.get("bot_id"),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

#PDF ENDPOINTS 

@app.post("/api/chat/{bot_id}/pdf")
async def chat_with_pdf(
    bot_id: str,
    pdf: UploadFile = File(...),
    message: Optional[str] = Form("What information is in this document?"),
    user_id: Optional[str] = Form(None)
):
    """
    Send PDF to bot for analysis
    Accepts PDF files
    """
    try:
        # Read PDF file
        pdf_bytes = await pdf.read()
        
        # Process with bot
        result = await handle_multimodal_request(
            bot_id,
            "pdf",
            message=message,
            pdf_data=pdf_bytes,
            user_id=user_id
        )
        
        return {
            "response": result.get("response"),
            "pdf_analysis": result.get("pdf_analysis", ""),
            "pdf_summary": result.get("pdf_summary"),
            "page_count": result.get("page_count"),
            "status": result.get("status"),
            "bot_name": result.get("bot_name"),
            "bot_id": result.get("bot_id"),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"PDF endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# BASE64 ENDPOINTS (for mobile/web)

class Base64ImageRequest(BaseModel):
    image_base64: str
    message: Optional[str] = "What is this product?"
    user_id: Optional[str] = None

class Base64AudioRequest(BaseModel):
    audio_base64: str
    user_id: Optional[str] = None
    language: str = "en"

class Base64PDFRequest(BaseModel):
    pdf_base64: str
    message: Optional[str] = "What information is in this document?"
    user_id: Optional[str] = None

@app.post("/api/chat/{bot_id}/image/base64")
async def chat_with_image_base64(bot_id: str, request: Base64ImageRequest):
    """Send image as base64 string"""
    try:
        # Decode base64
        image_bytes = base64.b64decode(request.image_base64)
        
        result = await handle_multimodal_request(
            bot_id,
            "image",
            message=request.message,
            image_data=image_bytes,
            user_id=request.user_id
        )
        
        return {
            "response": result.get("response"),
            "image_analysis": result.get("image_analysis", ""),
            "status": result.get("status"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 image processing failed: {str(e)}")

@app.post("/api/chat/{bot_id}/voice/base64")
async def chat_with_voice_base64(bot_id: str, request: Base64AudioRequest):
    """Send audio as base64 string"""
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        
        result = await handle_multimodal_request(
            bot_id,
            "voice",
            audio_data=audio_bytes,
            user_id=request.user_id,
            language=request.language
        )
        
        return {
            "response": result.get("response"),
            "transcribed_text": result.get("transcribed_text", ""),
            "status": result.get("status"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 audio processing failed: {str(e)}")

@app.post("/api/chat/{bot_id}/pdf/base64")
async def chat_with_pdf_base64(bot_id: str, request: Base64PDFRequest):
    """Send PDF as base64 string"""
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
        
        result = await handle_multimodal_request(
            bot_id,
            "pdf",
            message=request.message,
            pdf_data=pdf_bytes,
            user_id=request.user_id
        )
        
        return {
            "response": result.get("response"),
            "pdf_analysis": result.get("pdf_analysis", ""),
            "status": result.get("status"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 PDF processing failed: {str(e)}")

# UTILITY ENDPOINTS 

@app.get("/api/health")
async def health_check():
    """Health check endpoint with bot status"""
    bot_status = {}
    for bot_id, bot in bots.items():
        bot_status[bot_id] = {
            "name": bot.name,
            "status": "healthy",
            "capabilities": get_capabilities(bot),
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
    """List all available bots with their capabilities"""
    return {
        "bots": [
            {
                "id": bot_id,
                "name": bot.name,
                "description": getattr(bot, 'description', 'No description'),
                "capabilities": get_capabilities(bot)
            }
            for bot_id, bot in bots.items()
        ]
    }

@app.get("/api/bots/{bot_id}")
async def get_bot_info(bot_id: str):
    """Get detailed information about a specific bot"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    
    bot = bots[bot_id]
    return {
        "id": bot_id,
        "name": bot.name,
        "description": getattr(bot, 'description', 'No description'),
        "capabilities": get_capabilities(bot),
        "has_multimodal": {
            "voice": hasattr(bot, 'chat_with_voice'),
            "image": hasattr(bot, 'chat_with_image'),
            "pdf": hasattr(bot, 'chat_with_pdf')
        }
    }

# WEBSOCKET SUPPORT 

from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, bot_id: str):
        await websocket.accept()
        if bot_id not in self.active_connections:
            self.active_connections[bot_id] = []
        self.active_connections[bot_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, bot_id: str):
        if bot_id in self.active_connections:
            self.active_connections[bot_id].remove(websocket)
    
    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{bot_id}")
async def websocket_endpoint(websocket: WebSocket, bot_id: str):
    """WebSocket endpoint for real-time chat"""
    if bot_id not in bots:
        await websocket.close(code=1008, reason=f"Bot '{bot_id}' not found")
        return
    
    await manager.connect(websocket, bot_id)
    bot = bots[bot_id]
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Process message
            result = await bot.chat(
                message=data,
                user_id="websocket_user",
                conversation_history=None
            )
            
            await manager.send_message(result["response"], websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, bot_id)

# TEMPLATE FOR NEW BOTS 

BOT_TEMPLATE = """
# Template for creating a new bot with multimodal support

from typing import Optional, List, Dict, Any
from datetime import datetime

class YourNewBot:
    def __init__(self):
        self.bot_id = "your_bot_id"
        self.name = "Your Bot Name"
        self.description = "Bot description"
        
        # Initialize your clients here
        # self.groq_client = ...
        # self.supabase = ...
    
    # REQUIRED: Text chat method
    async def chat(self, message: str, user_id: Optional[str] = None,
                   conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        \"\"\"Main text chat method - REQUIRED\"\"\"
        # Your implementation here
        return {
            "response": "Bot response",
            "status": "success",
            "bot_name": self.name,
            "bot_id": self.bot_id,
            "timestamp": datetime.now().isoformat()
        }
    
    # OPTIONAL: Voice support
    async def chat_with_voice(self, audio_data, user_id: Optional[str] = None,
                              language: str = "en") -> Dict[str, Any]:
        \"\"\"Voice chat support - OPTIONAL\"\"\"
        # Transcribe audio and respond
        pass
    
    # OPTIONAL: Image support
    async def chat_with_image(self, message: str, image_data, user_id: Optional[str] = None,
                              conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        \"\"\"Image chat support - OPTIONAL\"\"\"
        # Analyze image and respond
        pass
    
    # OPTIONAL: PDF support
    async def chat_with_pdf(self, message: str, pdf_data, user_id: Optional[str] = None,
                            conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        \"\"\"PDF chat support - OPTIONAL\"\"\"
        # Analyze PDF and respond
        pass

# To add your bot to the server, simply import and add to the 'bots' dict:
# from bots.your_bot import YourNewBot
# bots["your_bot"] = YourNewBot()
"""

# RUN SERVER

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print(" MULTI-BOT GATEWAY SERVER v2.0")
    print("="*60)
    print(f" Server: http://0.0.0.0:8000")
    print(f" API Docs: http://localhost:8000/docs")
    print(f" Available bots: {', '.join(bots.keys())}")
    print("\n Bot Capabilities:")
    for bot_id, bot in bots.items():
        caps = get_capabilities(bot)
        print(f"   • {bot.name} ({bot_id}): {', '.join(caps)}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)