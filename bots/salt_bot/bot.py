# bots/salt_bot/bot.py
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

# Load salt-specific environment variables
load_dotenv("bots/salt_bot/.env")

class SaltBot:
    def __init__(self):
        self.bot_id = "salt"
        self.name = "SaltMate"
        self.description = "AI chatbot for Himalayan Salt Products"
        
        # Initialize Salt-specific clients
        self.groq_client = Groq(api_key=os.getenv("SALT_GROQ_API_KEY"))
        self.supabase: Client = create_client(
            os.getenv("SALT_SUPABASE_URL"),
            os.getenv("SALT_SUPABASE_KEY")
        )
        
        # Load static documentation
        self.documents = self.load_document_context()
        
        print(f"✅ {self.name} initialized")
    
    def load_document_context(self) -> Dict[str, str]:
        """Load static documentation about salt products"""
        return {
            "company_info": """
                Salt web is Pakistan's leading exporter of Himalayan salt products since 2010.
                We serve customers in over 50 countries worldwide with premium quality salt products.
            """,
            "product_catalog": """
                PRODUCTS:
                1. Himalayan Salt Lamps: 5-10kg hand-carved lamps with wooden base and dimmer cord. Price: $29.99-$49.99
                2. Edible Pink Salt: Fine and coarse grind. Available in 500g, 1kg, 5kg, 25kg packages. Price: $5.99/lb
                3. Salt Licks for Animals: 2kg blocks for horses, cattle, goats, and sheep. Price: $12.99/block
                4. Bath Salts: Therapeutic grade with 5 scents (lavender, eucalyptus, peppermint, rose, unscented). Price: $15.99/lb
                5. Salt Tiles: For saunas, spas, and wall decoration. Sizes: 8x4x2 inches. Price: $8.99/tile
                6. Cooking Slabs: Himalayan salt blocks for grilling and serving. Price: $39.99
            """,
            "shipping_policy": """
                SHIPPING POLICY:
                - Domestic (Pakistan): 3-5 business days, ₨500 flat rate
                - International: 10-15 business days via DHL/FedEx
                - Free shipping on orders over $100 worldwide
                - Tracking number provided for all orders
                - Bulk orders (50+ units) get priority shipping
            """,
            "return_policy": """
                RETURN POLICY:
                - 30-day money-back guarantee
                - Customer pays return shipping unless product is defective
                - Contact support@apexuniversal.com for RMA number
                - Refunds processed within 5-7 business days
                - Damaged items must be reported within 48 hours of delivery
            """,
            "faqs": """
                FREQUENTLY ASKED QUESTIONS:
                Q: Are your lamps genuine Himalayan salt?
                A: Yes, all lamps are sourced directly from the Khewra Salt Mine in Punjab, Pakistan.
                
                Q: Do you offer bulk/wholesale discounts?
                A: Yes! 10% off orders over 50 units, 15% off orders over 100 units.
                
                Q: How long do salt lamps last?
                A: With proper care, your salt lamp can last for years. Just keep it away from moisture.
                
                Q: Can I get custom packaging?
                A: Yes, custom packaging is available for bulk orders of 100+ units.
                
                Q: Do you ship to my country?
                A: We ship worldwide! Contact us for specific country shipping rates.
                
                Q: Are your products organic/certified?
                A: Yes, our edible salt is certified organic and non-GMO.
            """,
            "benefits": """
                BENEFITS OF HIMALAYAN SALT:
                - Contains 84 trace minerals
                - Improves air quality when used in lamps
                - Promotes better sleep
                - Reduces stress and anxiety
                - Natural air purifier
                - Helps with respiratory issues
                - Boosts energy levels
            """
        }
    
    async def get_product_info(self, product_name: str = None) -> Dict:
        """Query products from database"""
        try:
            query = self.supabase.table("products").select(
                "id, name, price, stock_quantity, category, description, image_folder, is_active"
            ).eq("is_active", True)
            
            if product_name:
                query = query.ilike("name", f"%{product_name}%")
            
            results = query.limit(5).execute()
            
            if results.data:
                return {
                    "type": "product_data",
                    "data": [
                        {
                            "name": p.get("name"),
                            "price": p.get("price"),
                            "stock": p.get("stock_quantity", 0),
                            "category": p.get("category"),
                            "description": (p.get("description") or "")[:200],
                            "has_images": bool(p.get("image_folder"))
                        }
                        for p in results.data
                    ]
                }
            return {}
        except Exception as e:
            print(f"Salt bot DB error: {e}")
            return {}
    
    async def get_user_orders(self, user_id: str, limit: int = 3) -> Dict:
        """Query user's orders from database"""
        if not user_id:
            return {}
        
        try:
            results = self.supabase.table("orders")\
                .select("id, status, total_amount, created_at, shipping_address")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            if results.data:
                enriched_orders = []
                for order in results.data:
                    items_result = self.supabase.table("order_items")\
                        .select("product_id, quantity, unit_price")\
                        .eq("order_id", order["id"])\
                        .execute()
                    
                    enriched_orders.append({
                        "order_id": str(order["id"])[:8],
                        "status": order.get("status", "pending"),
                        "total": order.get("total_amount"),
                        "date": order.get("created_at"),
                        "shipping_address": order.get("shipping_address", "Not provided"),
                        "item_count": len(items_result.data) if items_result.data else 0
                    })
                
                return {"type": "order_data", "data": enriched_orders}
            return {}
        except Exception as e:
            print(f"Salt bot orders error: {e}")
            return {}
    
    def classify_intent(self, message: str) -> str:
        """Classify user intent based on keywords"""
        message_lower = message.lower()
        
        intents = {
            "product_query": ["product", "lamp", "salt", "price", "cost", "available", "stock", "buy"],
            "order_query": ["order", "track", "delivery", "shipping status", "where is my"],
            "shipping_policy": ["shipping", "delivery time", "how long", "delivery cost", "ship to"],
            "return_policy": ["return", "refund", "exchange", "damaged", "defective"],
            "bulk_order": ["wholesale", "bulk", "large order", "quantity discount", "business"],
            "benefits": ["benefit", "advantage", "good for", "help with", "improve"],
            "faq": ["faq", "question", "how to", "can i", "do you", "is it"]
        }
        
        for intent, keywords in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        return "general"
    
    async def retrieve_context(self, message: str, user_id: Optional[str] = None) -> tuple[str, List[str]]:
        """Retrieve relevant context for the query"""
        context_parts = []
        context_sources = []
        intent = self.classify_intent(message)
        
        # Always include company info
        context_parts.append(self.documents["company_info"])
        context_sources.append("company_info")
        
        # Intent-specific context
        if intent in ["product_query", "general"]:
            context_parts.append(self.documents["product_catalog"])
            context_sources.append("product_catalog")
            
            product_data = await self.get_product_info()
            if product_data:
                context_parts.append(f"CURRENT PRODUCTS:\n{json.dumps(product_data['data'], indent=2)}")
                context_sources.append("database_products")
        
        if intent == "order_query" and user_id:
            order_data = await self.get_user_orders(user_id)
            if order_data:
                context_parts.append(f"USER'S ORDERS:\n{json.dumps(order_data['data'], indent=2)}")
                context_sources.append("user_orders")
        
        if intent in ["shipping_policy", "general"]:
            context_parts.append(self.documents["shipping_policy"])
            context_sources.append("shipping_policy")
        
        if intent == "return_policy":
            context_parts.append(self.documents["return_policy"])
            context_sources.append("return_policy")
        
        if intent == "bulk_order":
            context_parts.append(self.documents["faqs"])
            context_sources.append("faqs")
        
        if intent == "benefits":
            context_parts.append(self.documents["benefits"])
            context_sources.append("benefits")
        
        if intent in ["faq", "general"]:
            context_parts.append(self.documents["faqs"])
            context_sources.append("faqs")
        
        return "\n\n---\n\n".join(context_parts), context_sources
    
    async def generate_response(self, message: str, context: str, history: List[Dict] = None) -> str:
        """Generate response using Groq"""
        system_prompt = f"""You are SaltMate, a friendly customer support AI for Himalayan salt products.

CONTEXT:
{context}

Be warm, helpful, and concise (2-5 sentences). Use emojis occasionally. Never make up information."""
        
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
            return "I'm having trouble right now. Please try again."
    
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
            return {
                "response": f"Error: {str(e)}",
                "status": "error",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "timestamp": datetime.now().isoformat()
            }