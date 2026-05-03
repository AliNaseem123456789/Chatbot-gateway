# scripts/index_products_to_qdrant.py (Qdrant Cloud Version)
import os
import sys
import json
import base64
import asyncio
import requests
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from groq import Groq
from supabase import create_client
from PIL import Image
import io

load_dotenv("bots/ecommerce_bot/.env")

class ProductImageIndexer:
    """Indexer for Qdrant Cloud with local product images"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("PRODUCT IMAGE INDEXER (Qdrant Cloud)")
        print("="*70)        
        self.groq_client = Groq(api_key=os.getenv("ECOM_GROQ_API_KEY"))
        print("Groq initialized")
        qdrant_url = os.getenv("QDRANT_CLOUD_URL")  # e.g., "https://xyz.cloud.qdrant.io"
        qdrant_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            print("ERROR: Qdrant Cloud credentials missing in .env file!")
            print("   Please add:")
            print("   QDRANT_CLOUD_URL=https://your-cluster.cloud.qdrant.io")
            print("   QDRANT_CLOUD_API_KEY=your-api-key")
            sys.exit(1)
        
        self.qdrant_client = QdrantClient(
            url=qdrant_url,  
            api_key=qdrant_api_key,  
            timeout=60,  
        )
        print(f"Qdrant Cloud connected: {qdrant_url}")
        
        self.supabase = create_client(
            os.getenv("ECOM_SUPABASE_URL"),
            os.getenv("ECOM_SUPABASE_KEY")
        )
        print("Supabase initialized")
        
        self.image_base_path = Path(r"E:\projects_2\ecommerce-app\src\assets\products")
        print(f"Image path: {self.image_base_path}")
        
        self.collection_name = "ecommerce"  # Changed from "ecommerce_products"
        self.vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_size = 384
            print(f"Sentence Transformer loaded (384-dim vectors)")
        except ImportError:
            print(" Installing sentence-transformers...")
            os.system("pip install sentence-transformers")
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_size = 384
    
    def get_product_images(self, product_id: int) -> List[Path]:
        """Get all images for a product based on naming convention"""
        images = []
        for img_path in self.image_base_path.glob(f"{product_id}-*.jpeg"):
            images.append(img_path)
        for img_path in self.image_base_path.glob(f"{product_id}-*.jpg"):
            images.append(img_path)
        return sorted(images)
    
    async def extract_image_features(self, image_path: Path, product_name: str) -> Dict:
        """Extract features from a single image using Groq vision"""
        try:
            print(f"  Analyzing: {image_path.name}")
            
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Compress if needed
            if len(image_bytes) > 4 * 1024 * 1024:
                print(f"   Compressing image...")
                img = Image.open(io.BytesIO(image_bytes))
                if img.size[0] > 1024 or img.size[1] > 1024:
                    img.thumbnail((1024, 1024))
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
            
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this product image for a product called "{product_name}".
Extract features as JSON:
{{
    "product_type": "category (laptop/phone/headphones/etc)",
    "brand": "brand name",
    "colors": ["color1", "color2"],
    "features": ["feature1", "feature2", "feature3"],
    "design_style": "modern/classic/gaming/professional",
    "target_audience": "business/gaming/students/general",
    "price_segment": "budget/mid-range/premium/luxury",
    "what_it_is": "brief description of what this product looks like"
}}

Return ONLY valid JSON."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
            
            completion = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_tokens=500,
                temperature=0.2
            )
            
            analysis_text = completion.choices[0].message.content
            analysis_text = analysis_text.replace('```json', '').replace('```', '').strip()
            features = json.loads(analysis_text)
            
            return features
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    async def process_multiple_images(self, product_id: int, product_name: str, images: List[Path]) -> Dict:
        """Process multiple images for a single product and combine features"""
        all_features = []
        
        for img_path in images[:3]:
            features = await self.extract_image_features(img_path, product_name)
            if features:
                all_features.append(features)
        
        if not all_features:
            return None
        
        combined = {
            "product_type": all_features[0].get("product_type", ""),
            "brand": all_features[0].get("brand", ""),
            "colors": set(),
            "features": set(),
            "design_style": all_features[0].get("design_style", ""),
            "target_audience": all_features[0].get("target_audience", ""),
            "price_segment": all_features[0].get("price_segment", ""),
            "what_it_is": all_features[0].get("what_it_is", "")
        }
        
        for features in all_features:
            combined["colors"].update(features.get("colors", []))
            combined["features"].update(features.get("features", []))
        
        combined["colors"] = list(combined["colors"])
        combined["features"] = list(combined["features"])
        
        return combined
    
    async def create_text_embedding(self, text: str) -> List[float]:
        """Convert text to vector embedding"""
        embedding = self.embedder.encode(text).tolist()
        return embedding
    
    async def create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist in cloud"""
        try:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                print(f"\n Collection '{self.collection_name}' already exists")
                print(f"   Points: {collection_info.points_count}")
                print(f"   Vectors: {collection_info.vectors_count}")
                return True
            except Exception as e:
                # Collection doesn't exist, create it
                print(f"\n Creating collection '{self.collection_name}' in Qdrant Cloud...")
                
                # Create the collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                    # Optional: Configure for cloud performance
                    optimizers_config={
                        "default_segment_number": 2,
                        "indexing_threshold": 10000,
                    }
                )
                
                print(f"Collection '{self.collection_name}' created successfully!")
                print(f"   Vector size: {self.vector_size}")
                print(f"   Distance metric: COSINE")
                return True
                
        except Exception as e:
            print(f"Error with collection: {e}")
            print("\n Troubleshooting:")
            print("   1. Check your Qdrant Cloud URL is correct")
            print("   2. Verify your API key has write permissions")
            print("   3. Make sure your cluster is active")
            return False
    
    async def index_all_products(self):
        """Index all products from Supabase to Qdrant Cloud"""
        print("\nFetching products from Supabase...")
        
        products = self.supabase.table("products")\
            .select("product_id, name, description, price, stock, avg_rating, category_id")\
            .execute()
        
        print(f"Found {len(products.data)} products\n")
        
        points = []
        successful = 0
        failed = 0
        
        for idx, product in enumerate(products.data):
            print(f"\n{'='*50}")
            print(f"[{idx+1}/{len(products.data)}] Product ID {product['product_id']}: {product['name']}")
            print(f"{'='*50}")
            
            # Find images for this product
            images = self.get_product_images(product['product_id'])
            
            if not images:
                print(f"  No images found for product ID {product['product_id']}")
                failed += 1
                continue
            
            print(f"  Found {len(images)} image(s): {', '.join([img.name for img in images])}")
            
            # Analyze images
            features = await self.process_multiple_images(
                product['product_id'], 
                product['name'], 
                images
            )
            
            if not features:
                print(f"  Failed to extract features")
                failed += 1
                continue
            
            print(f"  Extracted: {features.get('product_type', 'Unknown')} - {features.get('brand', 'Unknown')}")
            print(f"  Colors: {', '.join(features.get('colors', []))}")
            print(f" Features: {', '.join(features.get('features', [])[:3])}")
            
            # Create searchable text
            searchable_text = f"""
            Product Name: {product['name']}
            Product Type: {features.get('product_type', '')}
            Brand: {features.get('brand', '')}
            Colors: {', '.join(features.get('colors', []))}
            Key Features: {', '.join(features.get('features', []))}
            Design Style: {features.get('design_style', '')}
            Target Audience: {features.get('target_audience', '')}
            Price Segment: {features.get('price_segment', '')}
            Price: ${product['price']}
            Description: {product['description']}
            What it looks like: {features.get('what_it_is', '')}
            """

            embedding = await self.create_text_embedding(searchable_text)            
            point = PointStruct(
                id=int(product['product_id']),
                vector=embedding,
                payload={
                    "product_id": product['product_id'],
                    "name": product['name'],
                    "description": product['description'],
                    "price": float(product['price']),
                    "stock": product['stock'],
                    "rating": float(product['avg_rating'] or 0),
                    "category_id": product['category_id'],
                    "image_count": len(images),
                    "first_image": str(images[0]) if images else None,
                    # Image features
                    "product_type": features.get('product_type', ''),
                    "brand": features.get('brand', ''),
                    "colors": features.get('colors', []),
                    "features": features.get('features', []),
                    "design_style": features.get('design_style', ''),
                    "target_audience": features.get('target_audience', ''),
                    "price_segment": features.get('price_segment', ''),
                    "indexed_at": datetime.now().isoformat()
                }
            )
            
            points.append(point)
            successful += 1
            
            # Save in batches of 5 to cloud
            if len(points) >= 5:
                print(f"\n    Saving batch of {len(points)} products to Qdrant Cloud...")
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"  Batch saved to cloud!")
                points = []
            
            await asyncio.sleep(0.5)        
        if points:
            print(f"\n   Saving final batch of {len(points)} products to Qdrant Cloud...")
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f" Final batch saved to cloud!")
        
        print(f"\n{'='*50}")
        print(f"INDEXING COMPLETE!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"   Total products: {len(products.data)}")
        print(f"    Collection: {self.collection_name}")
        print(f"{'='*50}")
    
    async def test_search(self, query: str):
        """Test search after indexing"""
        print(f"\n Testing search: '{query}'")
        
        # Create query vector
        query_vector = await self.create_text_embedding(query)
        
        # Search in cloud
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        print(f"\n Top results from Qdrant Cloud:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.payload['name']} (Score: {result.score:.3f})")
                print(f"   Type: {result.payload.get('product_type', 'Unknown')}")
                print(f"   Price: ${result.payload['price']}")
        else:
            print("   No results found")
    
    async def run(self):
        """Main indexing pipeline"""
        print("\n" + "="*70)
        print("STARTING PRODUCT IMAGE INDEXING PIPELINE (Qdrant Cloud)")
        print("="*70)
        
        print("\n Pipeline Configuration:")
        print(f"   Image storage: {self.image_base_path}")
        print(f"   Naming convention: {{product_id}}-{{n}}.jpeg")
        print(f"   Vector database: Qdrant Cloud")
        print(f"   Collection name: {self.collection_name}")
        print(f"   Vision model: {self.vision_model}")
        
        # Create collection if it doesn't exist
        if not await self.create_collection_if_not_exists():
            print(" Failed to setup collection. Exiting.")
            return
        
        # Index all products
        await self.index_all_products()
        
        # Show statistics
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"\n FINAL STATISTICS:")
            print(f"   Collection: {self.collection_name}")
            print(f"   Vector dimension: {self.vector_size}")
            print(f"   Points stored: {collection_info.points_count}")
            print(f"   Vectors stored: {collection_info.vectors_count}")
        except Exception as e:
            print(f"   Could not get stats: {e}")
        
        # Test search
        await self.test_search("silver laptop for business")
        await self.test_search("gaming headphones")
        
        print("\n INDEXING COMPLETE!")

if __name__ == "__main__":
    async def main():
        indexer = ProductImageIndexer()
        await indexer.run()
    
    asyncio.run(main())