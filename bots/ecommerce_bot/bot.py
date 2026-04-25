import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv("bots/ecommerce_bot/.env")

class EcommerceBot:
    def __init__(self):
        self.bot_id = "ecommerce"
        self.name = "ShopAssist"
        self.description = "AI customer support for electronics e-commerce store"
        
        # Initialize clients
        self.groq_client = Groq(api_key=os.getenv("ECOM_GROQ_API_KEY"))
        self.supabase: Client = create_client(
            os.getenv("ECOM_SUPABASE_URL"),
            os.getenv("ECOM_SUPABASE_KEY")
        )
        
        # Load static documentation
        self.documents = self.load_document_context()
        
        print(f"✅ {self.name} initialized")
    
    def load_document_context(self) -> Dict[str, str]:
        """Load static documentation about the e-commerce store"""
        return {
            "company_info": """
                ShopAssist Electronics is a leading online retailer of consumer electronics.
                We offer quality products with competitive prices and excellent customer service.
                Founded in 2020, we've served over 50,000 happy customers.
            """,
            "shipping_policy": """
                SHIPPING POLICY:
                - Free standard shipping on orders over $50
                - Standard delivery: 3-5 business days
                - Express delivery: 1-2 business days ($9.99)
                - Overnight delivery: Next day ($19.99)
                - Tracking number provided for all orders
            """,
            "return_policy": """
                RETURN POLICY:
                - 30-day hassle-free returns
                - Free returns for defective items
                - Customer pays return shipping for non-defective items (restocking fee 15%)
                - Refunds processed within 5-7 business days
            """,
            "payment_methods": """
                PAYMENT METHODS:
                - Credit/Debit Cards (Visa, Mastercard, Amex)
                - PayPal
                - Apple Pay
                - Google Pay
                - Buy Now, Pay Later (Klarna, Afterpay)
            """,
            "warranty": """
                WARRANTY:
                - 1-year manufacturer warranty on all products
                - Extended warranty available for purchase (2-3 years)
                - Free technical support for 90 days
            """,
            "faqs": """
                FREQUENTLY ASKED QUESTIONS:
                Q: How do I track my order?
                A: Use the tracking link sent to your email or check your order history.
                
                Q: Can I change or cancel my order?
                A: Yes within 1 hour of placing the order. Contact support immediately.
                
                Q: Do you price match?
                A: Yes, we price match with major competitors.
                
                Q: Are products genuine?
                A: Absolutely. We're authorized dealers for all brands.
                
                Q: Do you offer bulk discounts?
                A: Yes for business orders of 10+ units. Contact our sales team.
            """
        }
    
    async def get_categories(self) -> List[Dict]:
        """Get all product categories"""
        try:
            results = self.supabase.table("categories")\
                .select("category_id, name")\
                .execute()
            return results.data if results.data else []
        except Exception as e:
            print(f"Categories error: {e}")
            return []
    
    async def get_products(self, category_id: int = None, search: str = None, limit: int = 5) -> List[Dict]:
        """Query products from database"""
        try:
            query = self.supabase.table("products").select(
                "product_id, name, description, price, stock, avg_rating, category_id, categories(name)"
            )
            
            # Join with categories
            if category_id:
                query = query.eq("category_id", category_id)
            
            if search:
                query = query.ilike("name", f"%{search}%")
            
            # Only show in-stock products
            query = query.gt("stock", 0)
            
            results = query.limit(limit).execute()
            
            if results.data:
                return [
                    {
                        "id": p.get("product_id"),
                        "name": p.get("name"),
                        "description": (p.get("description") or "")[:150],
                        "price": float(p.get("price")),
                        "stock": p.get("stock", 0),
                        "rating": float(p.get("avg_rating") or 0),
                        "category": p.get("categories", {}).get("name") if p.get("categories") else None
                    }
                    for p in results.data
                ]
            return []
        except Exception as e:
            print(f"Products error: {e}")
            return []
    
    async def get_product_reviews(self, product_id: int, limit: int = 3) -> List[Dict]:
        """Get reviews for a specific product"""
        try:
            results = self.supabase.table("product_reviews")\
                .select("name, title, review, rating, created_at")\
                .eq("product_id", product_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            if results.data:
                return [
                    {
                        "name": r.get("name"),
                        "title": r.get("title"),
                        "review": (r.get("review") or "")[:100],
                        "rating": float(r.get("rating")),
                        "date": r.get("created_at")
                    }
                    for r in results.data
                ]
            return []
        except Exception as e:
            print(f"Reviews error: {e}")
            return []
    
    async def get_product_by_name(self, product_name: str) -> Dict:
        """Find product by name (fuzzy match)"""
        try:
            results = self.supabase.table("products")\
                .select("product_id, name, description, price, stock, avg_rating")\
                .ilike("name", f"%{product_name}%")\
                .limit(1)\
                .execute()
            
            if results.data:
                return results.data[0]
            return None
        except Exception as e:
            print(f"Product search error: {e}")
            return None
    
    async def get_user_orders(self, user_id: str, limit: int = 3) -> List[Dict]:
        """Get user's order history (if orders table exists)"""
        try:
            # Check if orders table exists
            results = self.supabase.table("orders")\
                .select("id, status, total_amount, created_at")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            if results.data:
                return [
                    {
                        "order_id": str(o.get("id"))[:8],
                        "status": o.get("status"),
                        "total": float(o.get("total_amount")),
                        "date": o.get("created_at")
                    }
                    for o in results.data
                ]
            return []
        except Exception:
            # Orders table might not exist yet
            return []
    
    def classify_intent(self, message: str) -> str:
        """Classify user intent based on keywords"""
        message_lower = message.lower()
        
        intents = {
            "product_search": ["product", "laptop", "phone", "computer", "headphone", "tablet", "monitor"],
            "price_check": ["price", "cost", "how much", "expensive", "cheap"],
            "availability": ["in stock", "available", "have", "got", "sell"],
            "reviews": ["review", "rating", "recommend", "good", "bad", "quality"],
            "shipping": ["shipping", "delivery", "arrive", "track", "shipped"],
            "returns": ["return", "refund", "exchange", "replace", "defective"],
            "payment": ["pay", "payment", "card", "paypal", "checkout"],
            "warranty": ["warranty", "guarantee", "protect", "cover"],
            "order_status": ["my order", "track order", "where is", "order status"],
            "categories": ["category", "type", "kind of", "what do you sell"]
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
        
        # Handle product search
        if intent in ["product_search", "price_check", "availability", "reviews"]:
            # Extract potential product name from message
            # Simple extraction - look for product keywords
            product_terms = ["laptop", "phone", "headphone", "tablet", "monitor", "keyboard", "mouse"]
            search_term = None
            for term in product_terms:
                if term in message.lower():
                    search_term = term
                    break
            
            # Get products
            products = await self.get_products(search=search_term, limit=5)
            if products:
                context_parts.append(f"CURRENT PRODUCTS:\n{json.dumps(products, indent=2)}")
                context_sources.append("database_products")
                
                # Get reviews for first product if asking about reviews
                if intent == "reviews" and products:
                    reviews = await self.get_product_reviews(products[0]["id"])
                    if reviews:
                        context_parts.append(f"PRODUCT REVIEWS:\n{json.dumps(reviews, indent=2)}")
                        context_sources.append("database_reviews")
        
        # Handle categories
        if intent == "categories":
            categories = await self.get_categories()
            if categories:
                context_parts.append(f"PRODUCT CATEGORIES:\n{json.dumps(categories, indent=2)}")
                context_sources.append("database_categories")
        
        # Handle orders
        if intent == "order_status" and user_id:
            orders = await self.get_user_orders(user_id)
            if orders:
                context_parts.append(f"USER ORDERS:\n{json.dumps(orders, indent=2)}")
                context_sources.append("user_orders")
            else:
                context_parts.append("User has no recent orders. Ask them to login or place an order.")
                context_sources.append("order_status_info")
        
        # Add policy documents based on intent
        policy_map = {
            "shipping": self.documents["shipping_policy"],
            "returns": self.documents["return_policy"],
            "payment": self.documents["payment_methods"],
            "warranty": self.documents["warranty"]
        }
        
        for intent_key, policy in policy_map.items():
            if intent == intent_key:
                context_parts.append(policy)
                context_sources.append(f"{intent_key}_policy")
        
        # Add FAQs for general queries
        if intent == "general":
            context_parts.append(self.documents["faqs"])
            context_sources.append("faqs")
        
        # Add shipping policy for order questions
        if intent == "order_status":
            context_parts.append(self.documents["shipping_policy"])
            context_sources.append("shipping_policy")
        
        return "\n\n---\n\n".join(context_parts), context_sources
    
    async def generate_response(self, message: str, context: str, history: List[Dict] = None) -> str:
        """Generate response using Groq"""
        system_prompt = f"""You are ShopAssist, a friendly customer support AI for an electronics e-commerce store.

CONTEXT INFORMATION:
{context}

YOUR ROLE:
- Help customers find products, check prices, answer shipping/return questions
- Be warm, helpful, and knowledgeable about electronics
- Provide accurate information from the context above
- If product is out of stock, suggest alternatives
- Keep responses concise (2-4 sentences)
- Use a friendly, professional tone with occasional emojis 🛒✨

IMPORTANT RULES:
- Never make up information not in context
- Always check database for product availability
- Suggest relevant products when appropriate
- Ask clarifying questions if needed
- End with offering additional help
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
            return "I'm having trouble right now. Please try again or contact support@shopassist.com"
    
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
                "response": "I apologize, but I encountered an error. Please try again or contact support.",
                "status": "error",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def search_products_api(self, query: str) -> List[Dict]:
        """Direct product search API endpoint"""
        return await self.get_products(search=query, limit=10)
    
    async def get_product_detail(self, product_id: int) -> Dict:
        """Get detailed product information"""
        try:
            # Get product
            product = self.supabase.table("products")\
                .select("*, categories(name)")\
                .eq("product_id", product_id)\
                .single()\
                .execute()
            
            if not product.data:
                return {"error": "Product not found"}
            
            # Get reviews
            reviews = await self.get_product_reviews(product_id, limit=5)
            
            return {
                "product": product.data,
                "reviews": reviews,
                "review_count": len(reviews),
                "avg_rating": product.data.get("avg_rating", 0)
            }
        except Exception as e:
            print(f"Product detail error: {e}")
            return {"error": str(e)}