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
import uuid
import warnings
# Or this (if . doesn't work):
from bots.ecommerce_bot.RagPipelines.MultimodalSupport.multimodal_processor import GroqOnlyMultimodalProcessor 
from bots.ecommerce_bot.RagPipelines.Rewriting.QueryRewriter import QueryRewritingPipeline
from bots.ecommerce_bot.RagPipelines.Intent.IntentClassifier import IntentClassifier, IntentType
# from logger import logging
from bots.ecommerce_bot.logger import setup_logger
load_dotenv("bots/ecommerce_bot/.env")

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
        self.logger = setup_logger("ecommerce_bot")
        
        # Initialize multimodal processor
        self.multimodal = GroqOnlyMultimodalProcessor(self.groq_client)
        self.query_pipeline = QueryRewritingPipeline(
            llm_client=self.groq_client,
            category_map={
                "laptops": ["laptop", "notebook", "macbook", "gaming laptop", "ultrabook"],
                "phones": ["phone", "smartphone", "iphone", "android", "pixel", "galaxy"],
                "audio": ["headphones", "earbuds", "speakers", "airpods", "headset"],
                "tablets": ["tablet", "ipad", "galaxy tab", "surface"],
                "accessories": ["case", "charger", "cable", "adapter", "stand"]
            }
        )
        # Load static documentation
        self.documents = self.load_document_context()
        
        # Initialize vector components
        self.qdrant_client = None
        self.embedder = None
        self._init_vector_components()
        
        # Chunk policies on startup
        self.chunk_policies()
        self.intent_classifier = IntentClassifier(llm_client=self.groq_client)

        
        print(f" {self.name} initialized with Groq multimodal support")
        print(f"   Vision Model: {self.multimodal.vision_model}")
        print(f"   Text Model: {self.multimodal.text_model}")
    
    def _init_vector_components(self):
        """Initialize Qdrant and embedder"""
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_CLOUD_URL"),
                api_key=os.getenv("QDRANT_CLOUD_API_KEY"),
                timeout=60,
            )
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("   Vector components initialized")
        except Exception as e:
            print(f"   Vector components error: {e}")
    
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
    
    def chunk_policies(self):
        """Chunk policy documents into vector store"""
        if not self.qdrant_client or not self.embedder:
            print("Vector components not available, skipping policy chunking")
            return
        
        print("Chunking policy documents...")
        
        policies = {
            "shipping": {
                "text": "Free shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping costs $9.99 and delivers in 1-2 business days. International shipping available to select countries.",
                "category": "shipping"
            },
            "returns": {
                "text": "30-day return policy. Free returns for defective items. 15% restocking fee for non-defective returns. Products must be in original condition. Refunds processed in 5-7 business days.",
                "category": "returns"
            },
            "warranty": {
                "text": "1-year manufacturer warranty on all electronics. Extended warranty available for purchase. Warranty covers manufacturing defects only. Does not cover accidental damage or normal wear.",
                "category": "warranty"
            },
            "payment": {
                "text": "We accept credit cards (Visa, Mastercard, American Express), PayPal, Apple Pay, Google Pay, and Klarna. All payments are securely processed.",
                "category": "payment"
            }
        }
        
        try:
            points = []
            for policy_name, policy_data in policies.items():
                sentences = policy_data["text"].split(". ")
                for i, sentence in enumerate(sentences):
                    if len(sentence) > 20:
                        embedding = self.embedder.encode(sentence).tolist()
                        from qdrant_client.models import PointStruct
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "policy_name": policy_name,
                                "category": policy_data["category"],
                                "text": sentence,
                                "chunk_index": i,
                                "type": "policy_chunk"
                            }
                        )
                        points.append(point)
            
            if points:
                self.qdrant_client.upsert(
                    collection_name="ecommerce_chunks",
                    points=points
                )
                print(f"   Indexed {len(points)} policy chunks")
        except Exception as e:
            print(f"   Policy chunking error: {e}")
    
    async def get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """Helper to get single product by ID"""
        try:
            result = self.supabase.table("products")\
                .select("product_id, name, description, price, stock, avg_rating")\
                .eq("product_id", product_id)\
                .single()\
                .execute()
            return result.data if result.data else None
        except:
            return None
    
    async def semantic_search_qdrant(self, search_query: str, limit: int = 5) -> List[Dict]:
        """Improved semantic search using vector embeddings"""
        if not self.qdrant_client or not self.embedder:
            return await self.get_products(search=search_query, limit=limit)
        
        try:
            query_vector = self.embedder.encode(search_query).tolist()
            
            results = self.qdrant_client.search(
                collection_name="ecommerce_products",
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.4
            )
            
            if not results:
                return []
            
            product_ids = [result.id for result in results]
            products = self.supabase.table("products")\
                .select("product_id, name, description, price, stock, avg_rating")\
                .in_("product_id", product_ids)\
                .execute()
            
            product_dict = {p['product_id']: p for p in products.data}
            enriched_results = []
            
            for result in results:
                if result.id in product_dict:
                    product = product_dict[result.id].copy()
                    product['similarity_score'] = result.score
                    product['search_type'] = 'semantic'
                    enriched_results.append(product)
            
            return enriched_results
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    async def hybrid_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Hybrid search combining vector and keyword results"""
        if not self.qdrant_client or not self.embedder:
            return await self.get_products(search=query, limit=limit)
        
        try:
            # Vector search
            query_vector = self.embedder.encode(query).tolist()
            vector_results = self.qdrant_client.search(
                collection_name="ecommerce_products",
                query_vector=query_vector,
                limit=limit * 2
            )
            
            # Keyword search
            keyword_results = await self.get_products(search=query, limit=limit * 2)
            
            # Combine and deduplicate
            combined = {}
            
            for r in vector_results:
                product = await self.get_product_by_id(r.id)
                if product:
                    combined[r.id] = {
                        **product,
                        "vector_score": r.score,
                        "keyword_score": 0,
                        "search_type": "hybrid"
                    }
            
            for p in keyword_results:
                if p['id'] in combined:
                    combined[p['id']]["keyword_score"] = 0.7
                    combined[p['id']]["final_score"] = combined[p['id']]["vector_score"] + 0.7
                else:
                    combined[p['id']] = {
                        **p,
                        "vector_score": 0,
                        "keyword_score": 0.5,
                        "final_score": 0.5
                    }
            
            for pid in combined:
                if "final_score" not in combined[pid]:
                    combined[pid]["final_score"] = combined[pid]["vector_score"] + combined[pid]["keyword_score"]
            
            sorted_results = sorted(combined.values(), key=lambda x: x.get("final_score", 0), reverse=True)
            return sorted_results[:limit]
            
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return await self.get_products(search=query, limit=limit)
    
    async def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank search results using Groq"""
        if len(results) <= 3:
            return results
        
        try:
            product_list = []
            for i, p in enumerate(results[:10]):
                product_list.append(f"{i+1}. {p['name']} - {p.get('description', '')[:100]}")
            
            rerank_prompt = f"""Query: "{query}"

Products:
{chr(10).join(product_list)}

Return ONLY the product numbers in order of relevance (most relevant first), separated by commas.
Example: "3,1,5,2,4"

Your ranking:"""
            
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": rerank_prompt}],
                temperature=0,
                max_tokens=50
            )
            
            ranking_text = completion.choices[0].message.content.strip()
            ranking = [int(x.strip()) - 1 for x in ranking_text.split(',') if x.strip().isdigit()]
            
            reranked = []
            for idx in ranking:
                if idx < len(results):
                    reranked.append(results[idx])
            
            for i, p in enumerate(results):
                if p not in reranked:
                    reranked.append(p)
            
            return reranked
            
        except Exception as e:
            print(f"Reranking error: {e}")
            return results
    
    async def get_relevant_policies(self, query: str) -> str:
        """Retrieve relevant policy chunks"""
        if not self.qdrant_client or not self.embedder:
            return ""
        
        try:
            query_vector = self.embedder.encode(query).tolist()
            
            results = self.qdrant_client.search(
                collection_name="ecommerce_chunks",
                query_vector=query_vector,
                limit=2,
                score_threshold=0.4
            )
            
            if results:
                policies = [r.payload.get('text', '') for r in results]
                return " ".join(policies)
            
        except Exception as e:
            print(f"Policy retrieval error: {e}")
        
        return ""
    async def rewrite_query(self, query: str, history: List[Dict] = None) -> str:
        """
        Enhanced query rewriting using the advanced pipeline.
        This replaces the old simple rewrite_query method.
        """
        try:
            result = await self.query_pipeline.rewrite(
                query=query,
                conversation_history=history,
                domain="ecommerce"
            )
            return result.rewritten
        except Exception as e:
            print(f"Advanced query rewrite error: {e}")
            # Fallback to original logic
            return await self._legacy_rewrite_query(query, history)
    
    # FIX: This method needs to be properly indented (inside the class)
    async def _legacy_rewrite_query(self, query: str, history: List[Dict] = None) -> str:
        """Fallback to original rewrite logic if pipeline fails"""
        if not history or len(history) < 2:
            return query
        
        try:
            recent_history = history[-4:] if len(history) > 4 else history
            history_text = []
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text.append(f"{role}: {msg['content']}")
            
            rewrite_prompt = f"""Conversation history:
{chr(10).join(history_text)}

Current query: "{query}"

If this query has pronouns (it, that, those, they, these, the second one) or is ambiguous, rewrite it to be self-contained.
If the query is clear, return it unchanged.

Rewritten query:"""
            
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.2,
                max_tokens=100
            )
            
            rewritten = completion.choices[0].message.content.strip()
            return rewritten if rewritten else query
            
        except Exception as e:
            print(f"Legacy query rewrite error: {e}")
            return query
    async def retrieve_context(self, message: str, user_id: Optional[str] = None) -> tuple[str, List[str]]:
        """Retrieve relevant context using vector embeddings"""
        context_parts = []
        sources = []
        
        # Vector product search
        if self.qdrant_client and self.embedder:
            try:
                query_vector = self.embedder.encode(message).tolist()
                results = self.qdrant_client.search(
                    collection_name="ecommerce_products",
                    query_vector=query_vector,
                    limit=3,
                    score_threshold=0.5
                )
                
                if results:
                    products = []
                    for result in results:
                        products.append({
                            "name": result.payload.get('name'),
                            "price": result.payload.get('price'),
                            "score": result.score
                        })
                    context_parts.append(f"RELEVANT PRODUCTS: {json.dumps(products, indent=2)}")
                    sources.append("vector_products")
            except Exception as e:
                print(f"Vector search error: {e}")
        
        # Fallback to keyword search
        if not sources:
            words = message.split()
            if words:
                search_term = words[-1] if len(words) > 0 else None
                if search_term and len(search_term) > 2:
                    products = await self.get_products(search=search_term, limit=3)
                    if products:
                        context_parts.append(f"PRODUCTS: {json.dumps(products, indent=2)}")
                        sources.append("keyword_products")
        
        # Add company info
        context_parts.append(f"COMPANY INFO: {self.documents['company_info']}")
        sources.append("company_info")
        
        return "\n\n---\n\n".join(context_parts), sources
    
    async def generate_response(self, message: str, context: str, history: List[Dict] = None) -> str:
        """Generate response using Groq"""
        system_prompt = f"""You are ShopAssist, a helpful e-commerce assistant.
CRITICAL RULES
1. ONLY mention products that are listed in the CONTEXT below
2. If a product is not in CONTEXT, DO NOT invent it
3. If no relevant products exist, say "I couldn't find matching products"
4. Do NOT make up prices, names, or specifications
5. Use EXACT product names from CONTEXT
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
    
    async def search_similar_products_qdrant(self, search_query: str, limit: int = 5) -> List[Dict]:
        """Legacy method - uses hybrid search now"""
        return await self.hybrid_search(search_query, limit)
    
    async def chat_with_image(self, message: str, image_data: Union[str, bytes, Path],
                         user_id: Optional[str] = None,
                         conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Chat with image using Qdrant vector search"""
        try:
            image_analysis = await self.multimodal.process_image_with_groq(image_data, message)
            analysis_json = json.loads(image_analysis)
            
            structured = analysis_json.get('structured', {})
            search_query = f"{structured.get('product_type', '')} {structured.get('brand', '')} {' '.join(structured.get('features', []))}"
            
            similar_products = await self.hybrid_search(search_query, limit=5)
            
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
            
            return await self.chat(transcribed_text, user_id, conversation_history)
            
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
            
            response_text = pdf_content.get('analysis', "I've reviewed your document.")
            
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
    
    async def chat(self, message: str, user_id: Optional[str] = None, 
                   conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Chat with intent classification and conditional rewriting"""
        
        self.logger.info(f"Processing message from user: {user_id or 'anonymous'}")
        self.logger.debug(f"Message content: {message[:200]}")        
        intent, method = await self.intent_classifier.classify(message)
        self.logger.info(f"Intent classified: {intent.value} (method={method})")        
        direct_response = self.intent_classifier.get_direct_response(intent)
        if direct_response:
            self.logger.info(f"Returning direct response for intent: {intent.value}")
            return {
                "response": direct_response,
                "status": "success",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "original_query": message,
                "intent": intent.value,
                "classification_method": method,
                "rag_used": False,
                "timestamp": datetime.now().isoformat()
                
            }
        
        # Step 3: Determine if query needs rewriting
        should_rewrite = self.intent_classifier.should_rewrite_query(intent)
        self.logger.info(f"Query rewriting required: {should_rewrite} for intent {intent.value}")
        
        # Step 4: Process based on intent
        if should_rewrite:
            self.logger.debug("Applying query rewriting pipeline")
            rewritten_result = await self.query_pipeline.rewrite(
                query=message,
                conversation_history=conversation_history,
                domain="ecommerce"
            )
            search_query = rewritten_result.rewritten
            sub_queries = rewritten_result.sub_queries
            constraints = rewritten_result.constraints
            techniques = rewritten_result.technique_used
            
            self.logger.info(f"Query rewritten: '{search_query}' (techniques: {techniques})")
            if constraints:
                self.logger.debug(f"Applied constraints: {constraints}")
        else:
            self.logger.debug(f"Skipping rewrite for low-intent query: {intent.value}")
            search_query = message
            sub_queries = [message]
            constraints = {}
            techniques = "none"
        
        # Step 5: Search based on intent type
        all_products = []
        if intent == IntentType.POLICY_QUESTION:
            self.logger.info(f"Retrieving policy information for query: {search_query}")
            policy_context = await self.get_relevant_policies(search_query)
            all_products = []
        else:
            self.logger.debug(f"Executing product search with {len(sub_queries)} sub-queries")
            for sub_query in sub_queries[:3]:
                products = await self.hybrid_search(sub_query, limit=3)
                all_products.extend(products)
                self.logger.debug(f"Sub-query '{sub_query}' returned {len(products)} products")
        
        # Step 6: Deduplicate and apply constraints
        unique_products = {}
        for p in all_products:
            pid = p.get('id') or p.get('product_id')
            if pid and pid not in unique_products:
                unique_products[pid] = p
        
        products_list = list(unique_products.values())
        self.logger.info(f"Retrieved {len(products_list)} unique products")
        
        # Apply constraints
        if constraints:
            original_count = len(products_list)
            if 'price_max' in constraints:
                products_list = [p for p in products_list if p.get('price', 0) <= constraints['price_max']]
            if 'min_rating' in constraints:
                products_list = [p for p in products_list if p.get('rating', 0) >= constraints['min_rating']]
            if original_count != len(products_list):
                self.logger.info(f"Filtered products: {original_count} → {len(products_list)} (constraints applied)")
        
        # Step 7: Rerank if products found
        if products_list:
            reranked_products = await self.rerank_results(search_query, products_list)
            self.logger.info(f"Reranked {len(reranked_products)} products")
        else:
            reranked_products = []
            self.logger.warning(f"No products found for query: {search_query}")
        
        # Step 8: Build context and response
        context_parts = []
        
        if reranked_products:
            context_parts.append(f"TOP PRODUCTS: {json.dumps(reranked_products[:3], indent=2)}")
        
        if intent == IntentType.POLICY_QUESTION:
            policy_context = await self.get_relevant_policies(search_query)
            if policy_context:
                context_parts.append(f"RELEVANT POLICIES: {policy_context}")
                self.logger.debug("Policy context added to response")
        
        context_parts.append(f"COMPANY INFO: {self.documents['company_info']}")
        context = "\n\n---\n\n".join(context_parts)
        
        response = await self.generate_response(search_query, context, conversation_history)
        self.logger.info(f"Response generated successfully ({len(response)} chars)")
        
        return {
            "response": response,
            "status": "success",
            "bot_name": self.name,
            "bot_id": self.bot_id,
            "original_query": message,
            "intent": intent.value,
            "classification_method": method,
            "rewritten_query": search_query if should_rewrite else "not rewritten",
            "techniques_used": techniques,
            "rag_used": should_rewrite,
            "products_found": len(reranked_products),
            "timestamp": datetime.now().isoformat(),
            "products": [  
        {
            "id": p.get("id") or p.get("product_id"),
            "name": p.get("name"),
            "price": p.get("price"),
            "rating": p.get("rating") or p.get("avg_rating")
        }
        for p in reranked_products[:5]
    ],
    "timestamp": datetime.now().isoformat()
        }
    async def search_products_api(self, query: str) -> List[Dict]:
        """Direct product search API endpoint using hybrid search"""
        return await self.hybrid_search(query, limit=10)
    
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