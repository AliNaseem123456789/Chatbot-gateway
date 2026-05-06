import os
import json
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

load_dotenv("bots/ecommerce_bot/.env")

class VectorIndexer:
    """Handles creating and updating vector embeddings from Supabase to Qdrant"""
    
    def __init__(self):
        # Initialize Supabase
        self.supabase: Client = create_client(
            os.getenv("ECOM_SUPABASE_URL"),
            os.getenv("ECOM_SUPABASE_KEY")
        )
        
        # Initialize Qdrant
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_CLOUD_URL"),
            api_key=os.getenv("QDRANT_CLOUD_API_KEY"),
            timeout=60,
        )        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
        self.collection_name = "ecommerce_products"
        self.chunk_collection = "ecommerce_chunks"
        
    def get_products_from_supabase(self, limit: Optional[int] = None) -> List[Dict]:
        """Fetch products from Supabase - using ACTUAL schema"""
        print(f"\n📦 Fetching products from Supabase...")
        
        try:
            # Query with correct column names from your schema
            query = self.supabase.table("products").select(
                "product_id, name, description, price, stock, avg_rating, category_id"
            )
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            
            if result.data:
                print(f"   ✅ Retrieved {len(result.data)} products")
                # Print first product as sample
                if result.data:
                    print(f"   📋 Sample: {result.data[0].get('name', 'N/A')}")
                return result.data
            else:
                print(f"   ⚠️ No products found in database!")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching products: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def fetch_category_name(self, category_id: int) -> str:
        """Fetch category name from categories table"""
        try:
            result = self.supabase.table("categories")\
                .select("name")\
                .eq("category_id", category_id)\
                .single()\
                .execute()
            
            if result.data:
                return result.data.get('name', '')
        except:
            pass
        return ""
    
    def create_product_text(self, product: Dict) -> str:
        """Create searchable text from product data - MATCHES YOUR SCHEMA"""
        text_parts = []
        
        # Core product info (all available in your schema)
        if product.get('name'):
            text_parts.append(product['name'])
        
        if product.get('description'):
            text_parts.append(product['description'])
        
        # Get category name if category_id exists
        if product.get('category_id'):
            category_name = self.fetch_category_name(product['category_id'])
            if category_name:
                text_parts.append(f"Category: {category_name}")
            else:
                text_parts.append(f"Category ID: {product['category_id']}")
        
        # Add pricing info
        if product.get('price'):
            text_parts.append(f"Price: ${product['price']}")
        
        # Add stock/availability
        if product.get('stock') is not None:
            if product['stock'] > 0:
                text_parts.append(f"In stock: {product['stock']} units")
            else:
                text_parts.append("Out of stock")
        
        # Add rating info
        if product.get('avg_rating') and product['avg_rating'] > 0:
            text_parts.append(f"Rating: {product['avg_rating']}/5 stars")
        
        return " ".join(text_parts) if text_parts else "No product information available"
    
    def create_collections(self):
        """Create Qdrant collections if they don't exist"""
        print("\n🔧 Creating/checking Qdrant collections...")
        
        try:
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]
            print(f"   Existing collections: {existing_names}")
            
            # Products collection
            if self.collection_name not in existing_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                )
                print(f"   ✅ Created collection: {self.collection_name}")
            else:
                print(f"   Collection {self.collection_name} already exists")
            
            # Chunks collection
            if self.chunk_collection not in existing_names:
                self.qdrant_client.create_collection(
                    collection_name=self.chunk_collection,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                )
                print(f"   ✅ Created collection: {self.chunk_collection}")
            else:
                print(f"   Collection {self.chunk_collection} already exists")
                
        except Exception as e:
            print(f"❌ Error creating collections: {e}")
    
    def index_products(self, batch_size: int = 50):
        """Index all products from Supabase to Qdrant"""
        print("\n" + "="*50)
        print("INDEXING PRODUCTS")
        print("="*50)
        
        products = self.get_products_from_supabase()
        
        if not products:
            print("❌ No products to index. Stopping product indexing.")
            return
        
        print(f"📦 Processing {len(products)} products...")
        
        # Process in batches
        successful = 0
        failed = 0
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i+batch_size]
            points = []
            
            for product in batch:
                try:
                    # Create searchable text
                    search_text = self.create_product_text(product)
                    
                    print(f"   🔍 Indexing: {product.get('name', 'Unknown')[:50]}...")
                    
                    # Generate embedding
                    embedding = self.embedder.encode(search_text).tolist()
                    
                    # Create point for Qdrant
                    point = PointStruct(
                        id=product['product_id'],  # Use product_id from schema
                        vector=embedding,
                        payload={
                            "product_id": product['product_id'],
                            "name": product.get('name', 'Unknown'),
                            "description": product.get('description', '')[:500],
                            "price": float(product.get('price', 0)) if product.get('price') else 0,
                            "stock": product.get('stock', 0),
                            "avg_rating": float(product.get('avg_rating', 0)) if product.get('avg_rating') else 0,
                            "category_id": product.get('category_id'),
                            "search_text": search_text[:1000]
                        }
                    )
                    points.append(point)
                    successful += 1
                    
                except Exception as e:
                    print(f"   ❌ Error processing product {product.get('product_id', 'unknown')}: {e}")
                    failed += 1
                    continue
            
            # Upsert to Qdrant
            if points:
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"   ✅ Indexed batch {i//batch_size + 1}/{(len(products)-1)//batch_size + 1} ({len(points)} products)")
                except Exception as e:
                    print(f"   ❌ Error upserting batch: {e}")
        
        print(f"\n📊 Indexing complete: {successful} successful, {failed} failed")
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks"""
        words = text.split()
        chunks = []
        
        if not words:
            return []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def index_policy_documents(self):
        """Index policy documents (returns, shipping, etc.) as chunks"""
        print("\n" + "="*50)
        print("INDEXING POLICY DOCUMENTS")
        print("="*50)
        
        policies = {
            "return_policy": """
                Our return policy allows returns within 30 days of purchase.
                Products must be in original condition with all accessories.
                Free returns for defective items.
                A 15% restocking fee applies for non-defective returns.
                Refunds are processed within 5-7 business days.
            """,
            "shipping_policy": """
                Free standard shipping on orders over $50.
                Standard shipping takes 3-5 business days.
                Express shipping costs $9.99, delivers in 1-2 business days.
                International shipping available to select countries.
                Tracking numbers provided via email.
            """,
            "warranty_policy": """
                1-year manufacturer warranty on all electronics.
                Extended warranty available for purchase.
                Warranty covers manufacturing defects only.
                Does not cover accidental damage or normal wear.
                Contact support for warranty claims.
            """,
            "payment_policy": """
                We accept credit cards (Visa, Mastercard, American Express).
                PayPal, Apple Pay, Google Pay also accepted.
                Klarna offers buy now, pay later options.
                All payments are securely processed.
            """
        }
        
        all_chunks = []
        
        for policy_name, policy_text in policies.items():
            chunks = self.chunk_document(policy_text.strip(), chunk_size=50, overlap=10)
            
            print(f"   📄 {policy_name}: {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embedder.encode(chunk).tolist()
                    
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "policy_name": policy_name,
                            "chunk_index": i,
                            "text": chunk,
                            "type": "policy"
                        }
                    )
                    all_chunks.append(point)
                except Exception as e:
                    print(f"   ❌ Error encoding chunk: {e}")
        
        # Batch upsert
        if all_chunks:
            try:
                self.qdrant_client.upsert(
                    collection_name=self.chunk_collection,
                    points=all_chunks
                )
                print(f"\n✅ Indexed {len(all_chunks)} policy chunks")
            except Exception as e:
                print(f"❌ Error upserting policy chunks: {e}")
        else:
            print("⚠️ No policy chunks to index")
    
    def verify_index(self):
        """Verify that data was actually indexed"""
        print("\n" + "="*50)
        print("VERIFYING INDEX")
        print("="*50)
        
        # Check products collection
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"📊 Products collection points: {collection_info.points_count}")
        except Exception as e:
            print(f"❌ Error checking products collection: {e}")
        
        # Check chunks collection
        try:
            collection_info = self.qdrant_client.get_collection(self.chunk_collection)
            print(f"📊 Chunks collection points: {collection_info.points_count}")
        except Exception as e:
            print(f"❌ Error checking chunks collection: {e}")
    
    def run_full_index(self):
        """Run complete indexing pipeline"""
        print("="*60)
        print("VECTOR INDEXING PIPELINE")
        print("="*60)
        
        # Create collections
        self.create_collections()
        
        # Index products
        self.index_products()
        
        # Index policy documents
        self.index_policy_documents()
        
        # Verify the index
        self.verify_index()
        
        print("\n" + "="*60)
        print("🎉 Full indexing complete!")
        print("="*60)


# Run the indexer
if __name__ == "__main__":
    print("Starting Vector Indexer...")
    indexer = VectorIndexer()
    indexer.run_full_index()