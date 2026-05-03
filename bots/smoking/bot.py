# bots/smoking_bot/bot.py
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv()

class SmokingBot:
    def __init__(self):
        self.bot_id = "smoking"
        self.name = "SmokeBuddy"
        self.description = "AI customer support for smoke shop wholesale products"
        
        # Initialize Groq client
        api_key = os.getenv("SMOKING_GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ Warning: SMOKING_GROQ_API_KEY not set")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=api_key)
        
        # Initialize Supabase client
        supabase_url = os.getenv("SMOKING_SUPABASE_URL")
        supabase_key = os.getenv("SMOKING_SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            print("Warning: Smoking Supabase credentials not set")
            self.supabase = None
        else:
            self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Load ONLY static policies and FAQs (NO product/brand data)
        self.documents = self.load_document_context()
        
        print(f"{self.name} initialized (database-first mode)")
    
    def load_document_context(self) -> Dict[str, str]:
        """Load ONLY static documentation - policies, FAQs, contact info"""
        return {
            "company_info": """
                SmokeBuddy Wholesale is a leading distributor of premium smoking products and accessories.
                We serve retail smoke shops, head shops, and convenience stores across the country.
                Established in 2018, we offer competitive wholesale pricing and fast shipping.
            """,
            
            "wholesale_info": """
                WHOLESALE INFORMATION:
                - Minimum Order: $500 for first-time wholesale customers
                - Bulk Discounts: 5-15% off based on quantity
                - Requirements: Valid business license and tax ID required
            """,
            
            "shipping_policy": """
                SHIPPING POLICY:
                - Free shipping on wholesale orders over $1,000
                - Standard delivery: 3-7 business days
                - Discreet packaging for all shipments
            """,
            
            "return_policy": """
                RETURN POLICY:
                - Defective products: 30-day replacement warranty
                - Unopened items: 15-day return window (15% restocking fee)
                - Must be 21+ to purchase tobacco/nicotine products
            """,
            
            "payment_methods": """
                PAYMENT METHODS:
                - Wire Transfer, ACH, Credit Card (3% fee)
                - Cryptocurrency (BTC, ETH) accepted
            """,
            
            "faqs": """
                FREQUENTLY ASKED QUESTIONS:
                Q: Do I need a license to buy wholesale? A: Yes, valid business license required.
                Q: What's your minimum wholesale order? A: $500 for first order.
                Q: Do you offer dropshipping? A: Yes, for qualified retailers.
                Q: Are products authentic? A: Yes, authorized distributor for all brands.
            """,
            
            "contact_info": """
                CONTACT: wholesale@smokebuddy.com | support@smokebuddy.com
                Phone: (555) 123-4567 | Hours: Mon-Fri 9am-6pm EST
            """
        }

    async def get_brands(self) -> List[Dict]:
        """Get all brands from database"""
        if not self.supabase:
            return []
        
        try:
            results = self.supabase.table("brands")\
                .select("id, name")\
                .order("name")\
                .execute()
            return results.data if results.data else []
        except Exception as e:
            print(f"Brands DB error: {e}")
            return []
    
    async def get_products(self, brand: str = None, category: str = None, 
                           search: str = None, limit: int = 10) -> List[Dict]:
        """Query products from database - LIVE DATA"""
        if not self.supabase:
            return []
        
        try:
            query = self.supabase.table("products").select(
                "id, title, brand, description, sku, categories, flavors, price, url"
            )
            
            if brand:
                query = query.eq("brand", brand)
            
            if category:
                query = query.contains("categories", [category])
            
            if search:
                query = query.ilike("title", f"%{search}%")
            
            results = query.limit(limit).execute()
            
            if results.data:
                return [
                    {
                        "id": p.get("id"),
                        "title": p.get("title"),
                        "brand": p.get("brand"),
                        "description": (p.get("description") or "")[:200],
                        "price": float(p.get("price")) if p.get("price") else None,
                        "categories": p.get("categories", []),
                        "flavors": p.get("flavors", []),
                        "sku": p.get("sku")
                    }
                    for p in results.data
                ]
            return []
        except Exception as e:
            print(f"Products DB error: {e}")
            return []
    
    async def get_products_by_brand(self, brand_name: str, limit: int = 10) -> List[Dict]:
        """Get all products for a specific brand - LIVE DATA"""
        return await self.get_products(brand=brand_name, limit=limit)
    
    async def get_product_by_sku(self, sku: str) -> Optional[Dict]:
        """Get product by SKU - LIVE DATA"""
        if not self.supabase:
            return None
        
        try:
            results = self.supabase.table("products")\
                .select("*")\
                .eq("sku", sku)\
                .limit(1)\
                .execute()
            
            if results.data:
                p = results.data[0]
                return {
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "brand": p.get("brand"),
                    "description": (p.get("description") or "")[:200],
                    "price": float(p.get("price")) if p.get("price") else None,
                    "categories": p.get("categories", []),
                    "flavors": p.get("flavors", []),
                    "sku": p.get("sku")
                }
            return None
        except Exception as e:
            print(f"Product SKU DB error: {e}")
            return None
    
    async def search_products_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Search products by category - LIVE DATA"""
        return await self.get_products(category=category, limit=limit)
    
    async def get_user_orders(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get user's order history - LIVE DATA"""
        if not self.supabase:
            return []        
        try:
            results = self.supabase.table("orders")\
                .select("id, status, total_amount, created_at, business_name")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            if results.data:
                return [
                    {
                        "order_id": o.get("id"),
                        "status": o.get("status", "pending"),
                        "total": float(o.get("total_amount") or 0),
                        "date": o.get("created_at"),
                        "business_name": o.get("business_name", "N/A")
                    }
                    for o in results.data
                ]
            return []
        except Exception as e:
            print(f"Orders DB error: {e}")
            return []
    
    def classify_intent(self, message: str) -> str:
        """Classify user intent based on keywords"""
        message_lower = message.lower()
        
        intents = {
            "product_search": ["product", "pipe", "vape", "paper", "grinder", "rig", "hookah", "cbd", "glass"],
            "brand_query": ["brand", "what brands", "which brands", "brands do you have"],
            "price_check": ["price", "cost", "how much", "expensive", "cheap"],
            "availability": ["in stock", "available", "have", "got"],
            "wholesale": ["wholesale", "bulk", "minimum order", "business license", "reseller"],
            "shipping": ["shipping", "delivery", "arrive", "track", "discreet"],
            "returns": ["return", "refund", "exchange", "defective", "damaged"],
            "order_status": ["my order", "track order", "where is", "order status"],
            "payment": ["pay", "payment", "card", "bitcoin", "crypto", "wire"],
            "faq": ["faq", "question", "how to", "can i", "license", "age", "21"],
            "contact": ["contact", "phone", "email", "support", "whatsapp"]
        }
        
        for intent, keywords in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return "general"
    
    # ============ CONTEXT RETRIEVAL (Hybrid: DB + Static) ============
    
    async def retrieve_context(self, message: str, user_id: Optional[str] = None) -> tuple[str, List[str]]:
        """Retrieve context - LIVE from DB, static for policies"""
        context_parts = []
        context_sources = []
        intent = self.classify_intent(message)
        
        # Always include company info (static)
        context_parts.append(self.documents["company_info"])
        context_sources.append("company_info")
        
        # ============ LIVE DATA FROM DATABASE ============
        
        # Get ALL brands from DB (live)
        brands = await self.get_brands()
        if brands:
            brand_names = [b["name"] for b in brands[:15]]
            context_parts.append(f"AVAILABLE BRANDS IN DATABASE:\n{', '.join(brand_names)}")
            context_sources.append("database_brands")
        
        # Product-related queries - get LIVE products from DB
        if intent in ["product_search", "price_check", "availability", "brand_query"]:
            # Extract brand mention from message
            mentioned_brand = None
            for brand in brands:
                if brand["name"].lower() in message.lower():
                    mentioned_brand = brand["name"]
                    break
            
            # Get products from DB
            products = await self.get_products(brand=mentioned_brand, limit=8)
            if products:
                context_parts.append(f"LIVE PRODUCTS FROM DATABASE:\n{json.dumps(products, indent=2)}")
                context_sources.append("database_products")
            else:
                context_parts.append("No products found matching your criteria in the database.")
                context_sources.append("database_empty")
        
        # Order status - LIVE from DB
        if intent == "order_status" and user_id:
            orders = await self.get_user_orders(user_id)
            if orders:
                context_parts.append(f"USER'S ORDERS FROM DATABASE:\n{json.dumps(orders, indent=2)}")
                context_sources.append("database_orders")
            else:
                context_parts.append("No orders found for this user in the database.")
                context_sources.append("database_empty")
        
        # ============ STATIC DOCUMENTS (Policies only) ============
        
        if intent in ["wholesale", "general"]:
            context_parts.append(self.documents["wholesale_info"])
            context_sources.append("wholesale_info")
        
        if intent in ["shipping", "general"]:
            context_parts.append(self.documents["shipping_policy"])
            context_sources.append("shipping_policy")
        
        if intent == "returns":
            context_parts.append(self.documents["return_policy"])
            context_sources.append("return_policy")
        
        if intent == "payment":
            context_parts.append(self.documents["payment_methods"])
            context_sources.append("payment_methods")
        
        if intent in ["faq", "general"]:
            context_parts.append(self.documents["faqs"])
            context_sources.append("faqs")
        
        if intent == "contact":
            context_parts.append(self.documents["contact_info"])
            context_sources.append("contact_info")
        
        return "\n\n---\n\n".join(context_parts), context_sources
    
    # ============ RESPONSE GENERATION ============
    
    async def generate_response(self, message: str, context: str, history: List[Dict] = None) -> str:
        """Generate response using Groq"""
        if not self.groq_client:
            return "I apologize, but the AI service is not configured. Please contact support@smokebuddy.com"
        
        system_prompt = f"""You are SmokeBuddy, a professional customer support AI for a smoke shop wholesale company.

IMPORTANT: The information below comes from TWO sources:
1. LIVE DATABASE DATA - Products, brands, orders (real, current data)
2. STATIC POLICIES - Shipping, returns, wholesale requirements

CONTEXT INFORMATION:
{context}

YOUR ROLE:
- Use LIVE database data for product/brand/order questions
- Use static policies for shipping/returns/wholesale questions
- Be honest: if no products match, say so and suggest alternatives
- Never share hardcoded product lists - only use database results
- Always mention age verification (21+) when discussing products
- Keep responses concise (2-4 sentences)

RULES:
- Never make up products not in database results
- Always check database for availability
- Offer additional help at the end of responses
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-5:])
        messages.append({"role": "user", "content": message})
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Groq error: {e}")
            return "I'm having trouble right now. Please email support@smokebuddy.com for assistance."
    
    # ============ MAIN CHAT METHOD ============
    
    async def chat(self, message: str, user_id: Optional[str] = None, 
                   conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Main chat method for the bot"""
        try:
            context, sources = await self.retrieve_context(message, user_id)
            response = await self.generate_response(message, context, conversation_history)
            
            return {
                "response": response,
                "status": "success",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "context_used": sources,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Chat error: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please contact support for assistance.",
                "status": "error",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "timestamp": datetime.now().isoformat()
            }