import os
import json
import base64
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import requests
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv("bots/ecommerce_bot/.env")

class GroqOnlyMultimodalProcessor:
    """Use Groq for all multimodal processing with current models"""
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.text_model = "llama-3.1-8b-instant"  # Fast text model
        self.vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"          
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    async def process_image_with_groq(self, image_data: Union[str, bytes, Path], 
                                    query: str = None) -> str:
        """Process image using Groq's Llama-4 Scout vision model"""
        try:
            if isinstance(image_data, str):
                if image_data.startswith(('http://', 'https://')):
                    response = requests.get(image_data, timeout=10)
                    response.raise_for_status()
                    image_data = response.content
                elif Path(image_data).exists():
                    with open(image_data, 'rb') as f:
                        image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are a product analyst. Look at this image and answer these questions in a natural way:

    1. What product do you see? (Be specific - laptop, smartphone, headphones, etc.)
    2. What color is it?
    3. Can you identify the brand? (Look for logos)
    4. What notable features do you see?
    5. What condition does it appear to be in?

    Customer asked: {query if query else 'What is this product?'}

    Write a helpful response describing what you see in the image. Be conversational and specific."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call Groq with vision model
            completion = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_tokens=500,
                temperature=0.4
            )
            
            response_text = completion.choices[0].message.content            
            structured_prompt = f"""
            Based on this description: "{response_text}"
            
            Extract the key information as a JSON object. Return ONLY valid JSON:
            {{
                "product_type": "main product category",
                "brand": "brand name or unknown",
                "color": "main color",
                "features": ["feature1", "feature2"],
                "condition": "new/used/like new",
                "description": "brief 1-sentence summary"
            }}
            """
            
            try:
                structured = self.groq_client.chat.completions.create(
                    model=self.text_model,
                    messages=[{"role": "user", "content": structured_prompt}],
                    temperature=0.2,
                    max_tokens=300
                )                
                json_str = structured.choices[0].message.content
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                analysis_json = json.loads(json_str)                
                return json.dumps({
                    "natural_description": response_text,
                    "structured": analysis_json
                })
                
            except:
                # If JSON parsing fails, just return the natural description
                return json.dumps({
                    "natural_description": response_text,
                    "structured": {
                        "product_type": "Unknown",
                        "brand": "Unknown", 
                        "color": "Unknown",
                        "features": [],
                        "condition": "Unknown",
                        "description": response_text[:200]
                    }
                })
            
        except Exception as e:
            print(f"Image processing error: {e}")
            return json.dumps({
                "error": str(e),
                "natural_description": "I couldn't process this image. Please describe the product you're looking for.",
                "structured": {
                    "product_type": "Unknown",
                    "brand": "Unknown",
                    "color": "Unknown",
                    "features": [],
                    "condition": "Unknown",
                    "description": "Unable to analyze image"
                }
            })
    
    async def process_audio_with_groq_ecosystem(self, audio_data: Union[str, bytes, Path], 
                                                 language: str = "en") -> str:
        """Transcribe audio using external service, then enhance with Groq"""
        
        # Try available STT services
        if self.deepgram_api_key:
            transcript = await self._transcribe_with_deepgram(audio_data, language)
       
        else:
            return "Voice transcription requires Deepgram, AssemblyAI, or HuggingFace API key."
        
        if transcript and transcript != "":
            # Enhance with Groq
            return await self._enhance_transcript_with_groq(transcript)
        return ""
    
    async def _enhance_transcript_with_groq(self, transcript: str) -> str:
        """Use Groq to clean up transcript"""
        if not transcript:
            return ""
        
        try:
            enhancement = self.groq_client.chat.completions.create(
                model=self.text_model,  # Using text model
                messages=[
                    {
                        "role": "system",
                        "content": """Clean up speech-to-text transcript for e-commerce.
                        Fix grammar, capitalize properly, correct product terms, remove filler words.
                        Return ONLY cleaned transcript, nothing else."""
                    },
                    {
                        "role": "user",
                        "content": f"Raw transcript: {transcript}"
                    }
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            return enhancement.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq enhancement error: {e}")
            return transcript
    
    async def _transcribe_with_deepgram(self, audio_data, language="en") -> str:
        """Use Deepgram for transcription"""
        try:
            if isinstance(audio_data, str) and Path(audio_data).exists():
                with open(audio_data, 'rb') as f:
                    audio_bytes = f.read()
            elif isinstance(audio_data, bytes):
                audio_bytes = audio_data
            else:
                response = requests.get(audio_data, timeout=10)
                audio_bytes = response.content
            
            print(f"Audio size: {len(audio_bytes)} bytes")
            
            if len(audio_bytes) < 1000:
                print("Audio too small - likely empty or invalid")
                return ""            
            with open("debug_audio.webm", "wb") as f:
                f.write(audio_bytes)
            print("Saved audio to debug_audio.webm for inspection")
            
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {self.deepgram_api_key}",
            }
            params = {
                "model": "nova-2",
                "language": language,
            }
            
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                data=audio_bytes,
                # headers={"Content-Type": "audio/webm"}
            )
            
            if response.status_code == 200:
                result = response.json()
                transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
                print(f"Deepgram transcript: '{transcript}'")
                return transcript
            else:
                print(f"Deepgram error {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            print(f"Deepgram error: {e}")
            return ""
   
    async def process_pdf_with_groq(self, pdf_data: Union[str, bytes, Path]) -> Dict:
        """Process PDF using Groq for analysis - handles any PDF"""
        try:
            import PyPDF2
            
            # Extract text (PyPDF2 is lightweight and free)
            if isinstance(pdf_data, str):
                if Path(pdf_data).exists():
                    pdf_path = pdf_data
                else:
                    response = requests.get(pdf_data, timeout=10)
                    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_pdf.write(response.content)
                    temp_pdf.close()
                    pdf_path = temp_pdf.name
            else:
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_pdf.write(pdf_data)
                temp_pdf.close()
                pdf_path = temp_pdf.name
            
            # Extract text
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages[:10]:  # First 10 pages
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
            
            full_text = "\n\n".join(text_content)
            
            # Use Groq for intelligent analysis - handles ANY PDF
            analysis = self.groq_client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful document analyzer. Analyze this PDF and provide a friendly response.

    FIRST, determine what type of document this is:
    - Product-related (invoice, warranty, manual, receipt, product specs) → Help with e-commerce
    - Resume/CV → Offer polite assistance
    - Personal document → Be respectful and helpful
    - Other document → Provide general assistance

    Based on the document type, respond appropriately:

    IF PRODUCT-RELATED:
    Extract: product_specs, pricing, warranty_terms, return_policy, dates

    IF NOT PRODUCT-RELATED (resume, personal, etc.):
    Provide a friendly message like:
    "I see you uploaded a [type of document]. While I'm specialized in helping with product-related documents, I can still help you find products or answer questions about our electronics store. Is there anything specific I can help you with today?"

    Return as JSON with:
    {
        "document_type": "product/resume/personal/other",
        "is_product_related": true/false,
        "analysis": "friendly response for the user",
        "extracted_data": {} // Only if product-related
    }"""
                    },
                    {
                        "role": "user",
                        "content": f"Document text (first 6000 chars):\n{full_text[:6000]}"
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Clean up
            if 'temp_pdf' in locals() and Path(pdf_path).exists():
                os.unlink(pdf_path)
            
            # Parse the response
            try:
                result = json.loads(analysis.choices[0].message.content)
                return {
                    "analysis": result.get("analysis", "I've analyzed your document."),
                    "document_type": result.get("document_type", "other"),
                    "is_product_related": result.get("is_product_related", False),
                    "extracted_data": result.get("extracted_data", {}),
                    "text_preview": full_text[:500],
                    "page_count": len(pdf_reader.pages) if 'pdf_reader' in locals() else 0
                }
            except:
                # Fallback if JSON parsing fails
                return {
                    "analysis": analysis.choices[0].message.content,
                    "document_type": "other",
                    "is_product_related": False,
                    "extracted_data": {},
                    "text_preview": full_text[:500],
                    "page_count": len(pdf_reader.pages) if 'pdf_reader' in locals() else 0
                }
                
        except Exception as e:
            print(f"PDF processing error: {e}")
            return {
                "error": str(e),
                "analysis": "I had trouble reading that PDF. Please make sure it's a valid document.",
                "document_type": "error",
                "is_product_related": False
            }


class EcommerceBot:
    def __init__(self):
        self.bot_id = "ecommerce"
        self.name = "ShopAssist"
        self.description = "AI customer support with Groq-powered multimodal features"
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("ECOM_GROQ_API_KEY"))
        self.supabase: Client = create_client(
            os.getenv("ECOM_SUPABASE_URL"),
            os.getenv("ECOM_SUPABASE_KEY")
        )
        
        # Initialize multimodal processor
        self.multimodal = GroqOnlyMultimodalProcessor(self.groq_client)
        
        # Load static documentation
        self.documents = self.load_document_context()
        
        print(f" {self.name} initialized with Groq multimodal support")
        print(f"   Vision Model: {self.multimodal.vision_model}")
        print(f"   Text Model: {self.multimodal.text_model}")
    
    def load_document_context(self) -> Dict[str, str]:
        """Load static documentation"""
        return {
            "company_info": "ShopAssist Electronics - quality electronics since 2020",
            "shipping_policy": "Free shipping over $50, standard 3-5 days, express $9.99",
            "return_policy": "30-day returns, free for defects, 15% restocking fee otherwise",
            "payment_methods": "Credit cards, PayPal, Apple Pay, Google Pay, Klarna",
            "warranty": "1-year manufacturer warranty, extended available",
            "faqs": "Common questions about tracking, cancellation, price matching"
        }
    async def search_similar_products_qdrant(self, search_query: str, limit: int = 5) -> List[Dict]:
        """Search Qdrant using text query - works with any qdrant-client version"""
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            
            # Connect to Qdrant Cloud
            qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_CLOUD_URL"),
                api_key=os.getenv("QDRANT_CLOUD_API_KEY"),
                timeout=60,
            )
            
            # Create embedding from search query
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            query_vector = embedder.encode(search_query).tolist()
            
            # Try different method names (universal approach)
            results = None
            
            # Method 1: Modern versions (v1.7+)
            if hasattr(qdrant_client, 'search'):
                results = qdrant_client.search(
                    collection_name="ecommerce",
                    query_vector=query_vector,
                    limit=limit
                )
            # Method 2: Older versions (v1.0 - v1.6)
            elif hasattr(qdrant_client, 'search_collection'):
                results = qdrant_client.search_collection(
                    collection_name="ecommerce",
                    query_vector=query_vector,
                    limit=limit
                )
            # Method 3: Points API
            elif hasattr(qdrant_client, 'query_points'):
                response = qdrant_client.query_points(
                    collection_name="ecommerce",
                    query=query_vector,
                    limit=limit
                )
                results = response.points
            else:
                print("No search method found in qdrant client")
                return []
            
            if not results:
                return []
            
            # Get full product details from Supabase
            product_ids = [result.id for result in results]
            if product_ids:
                products = self.supabase.table("products")\
                    .select("product_id, name, description, price, stock, avg_rating")\
                    .in_("product_id", product_ids)\
                    .execute()
                
                # Combine with scores
                for result in results:
                    for product in products.data:
                        if product['product_id'] == result.id:
                            product['similarity_score'] = result.score
                            break
                
                return products.data
            return []
            
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []
    
    async def chat_with_image(self, message: str, image_data: Union[str, bytes, Path],
                         user_id: Optional[str] = None,
                         conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Chat with image using Qdrant vector search"""
        try:
            # Step 1: Analyze image with Groq
            image_analysis = await self.multimodal.process_image_with_groq(image_data, message)
            analysis_json = json.loads(image_analysis)
            
            # Step 2: Create search query from analysis
            structured = analysis_json.get('structured', {})
            
            search_query = f"{structured.get('product_type', '')} {structured.get('brand', '')} {' '.join(structured.get('features', []))}"
            
            # Step 3: Search in Qdrant
            similar_products = await self.search_similar_products_qdrant(search_query, limit=5)
            
            # Step 4: Generate response
            if similar_products:
                response = " I found these products similar to your image:\n\n"
                for i, product in enumerate(similar_products[:3], 1):
                    response += f"{i}. **{product['name']}** - ${product['price']}\n"
                    response += f"    Rating: {product.get('avg_rating', 'N/A')}/5\n"
                    if product.get('similarity_score'):
                        response += f"    Match: {int(product['similarity_score'] * 100)}%\n"
                    response += "\n"
            else:
                response = f"I couldn't find exact matches. Could you describe what features you're looking for in a {structured.get('product_type', 'product')}?"
            
            return {
                "response": response,
                "status": "success",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "image_analysis": image_analysis,
                "similar_products": similar_products,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Image chat error: {e}")
            return {
                "response": "I couldn't process that image. Please try again.",
                "status": "error",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def chat_with_voice(self, audio_data: Union[str, bytes, Path],
                             user_id: Optional[str] = None,
                             language: str = "en",
                             conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Chat with voice input"""
        try:
            transcribed_text = await self.multimodal.process_audio_with_groq_ecosystem(audio_data, language)
            
            if not transcribed_text or "error" in transcribed_text.lower():
                return {
                    "response": "I couldn't understand the audio. Please try speaking clearly or type your message.",
                    "status": "error",
                    "bot_name": self.name,
                    "bot_id": self.bot_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            context, sources = await self.retrieve_context(transcribed_text, user_id)
            response = await self.generate_response(transcribed_text, context, conversation_history)
            
            return {
                "response": response,
                "transcribed_text": transcribed_text,
                "status": "success",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "context_used": sources,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Voice chat error: {e}")
            return {
                "response": "I had trouble processing your voice message. Please try again.",
                "status": "error",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def chat_with_pdf(self, message: str, pdf_data: Union[str, bytes, Path],
                       user_id: Optional[str] = None,
                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Chat with PDF using Groq for analysis"""
        try:
            pdf_content = await self.multimodal.process_pdf_with_groq(pdf_data)
            
            if "error" in pdf_content:
                return {
                    "response": f"I had trouble reading that PDF. Please try again with a different document.",
                    "status": "error",
                    "bot_name": self.name,
                    "bot_id": self.bot_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Use the analysis directly from the PDF processor
            response_text = pdf_content.get('analysis', "I've reviewed your document.")
            
            # If it's not product-related, add a helpful note
            if not pdf_content.get('is_product_related', False):
                response_text += "\n\nIs there anything specific about our electronics products I can help you with? 🛒"
            
            return {
                "response": response_text,
                "status": "success",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "pdf_analysis": pdf_content.get('analysis'),
                "document_type": pdf_content.get('document_type', 'other'),
                "pdf_summary": pdf_content.get('analysis', '')[:500],
                "page_count": pdf_content.get('page_count', 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"PDF chat error: {e}")
            return {
                "response": "I had trouble processing that PDF. Please ensure it's a valid document.",
                "status": "error",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "timestamp": datetime.now().isoformat()
            }
    async def get_categories(self) -> List[Dict]:
        """Get all product categories"""
        try:
            results = self.supabase.table("categories").select("category_id, name").execute()
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
            if category_id:
                query = query.eq("category_id", category_id)
            if search:
                query = query.ilike("name", f"%{search}%")
            query = query.gt("stock", 0)
            results = query.limit(limit).execute()
            
            if results.data:
                return [{
                    "id": p.get("product_id"),
                    "name": p.get("name"),
                    "description": (p.get("description") or "")[:150],
                    "price": float(p.get("price")),
                    "stock": p.get("stock", 0),
                    "rating": float(p.get("avg_rating") or 0),
                    "category": p.get("categories", {}).get("name") if p.get("categories") else None
                } for p in results.data]
            return []
        except Exception as e:
            print(f"Products error: {e}")
            return []
    
    async def retrieve_context(self, message: str, user_id: Optional[str] = None) -> tuple[str, List[str]]:
        """Retrieve relevant context"""
        context_parts = [self.documents["company_info"]]
        sources = ["company_info"]
        
        # Simple product search
        words = message.split()
        if words:
            search_term = words[-1] if len(words) > 0 else None
            if search_term and len(search_term) > 2:
                products = await self.get_products(search=search_term, limit=3)
                if products:
                    context_parts.append(f"PRODUCTS: {json.dumps(products, indent=2)}")
                    sources.append("products")
        
        return "\n\n---\n\n".join(context_parts), sources
    
    async def generate_response(self, message: str, context: str, history: List[Dict] = None) -> str:
        """Generate response using Groq"""
        system_prompt = f"""You are ShopAssist, a helpful e-commerce assistant.

CONTEXT: {context}

Rules:
- Be helpful and concise (2-3 sentences)
- Suggest products when relevant
- Be friendly and use emojis occasionally
- Never make up information

Response:"""
        
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-3:])
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
        """Standard text chat"""
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
    
    async def search_products_api(self, query: str) -> List[Dict]:
        """Direct product search API endpoint"""
        return await self.get_products(search=query, limit=10)
    
    async def get_product_detail(self, product_id: int) -> Dict:
        """Get detailed product information"""
        try:
            product = self.supabase.table("products")\
                .select("*, categories(name)")\
                .eq("product_id", product_id)\
                .single()\
                .execute()
            
            if not product.data:
                return {"error": "Product not found"}
            
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