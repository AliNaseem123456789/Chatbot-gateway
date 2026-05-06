# debug_text_index.py
import os
import sys
from dotenv import load_dotenv
from supabase import create_client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv("bots/ecommerce_bot/.env")

print("="*60)
print("DEBUG: TEXT INDEXING CHECK")
print("="*60)

# 1. Check Supabase connection
print("\n1. Checking Supabase...")
supabase = create_client(
    os.getenv("ECOM_SUPABASE_URL"),
    os.getenv("ECOM_SUPABASE_KEY")
)

# Fetch ONE product
result = supabase.table("products").select("*").limit(1).execute()
if result.data:
    print(f"   ✅ Supabase connected")
    print(f"   Sample product: {result.data[0]['name']}")
    print(f"   Product ID: {result.data[0]['product_id']}")
else:
    print(f"   ❌ No products found")
    sys.exit()

# 2. Check Qdrant connection
print("\n2. Checking Qdrant...")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_CLOUD_API_KEY"),
    timeout=60,
)

try:
    collections = qdrant_client.get_collections()
    print(f"   ✅ Qdrant connected")
    print(f"   Collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"   ❌ Qdrant error: {e}")
    sys.exit()

# 3. Check if 'ecommerce' collection exists
print("\n3. Checking 'ecommerce' collection...")
try:
    collection_info = qdrant_client.get_collection("ecommerce_products")
    print(f"   ✅ Collection 'ecommerce' exists")
    print(f"   Total points: {collection_info.points_count}")
except Exception as e:
    print(f"   ❌ Collection 'ecommerce' not found: {e}")
    sys.exit()

# 4. Check what's already in Qdrant for a specific product
product_id = result.data[0]['product_id']
print(f"\n4. Checking product ID {product_id} in Qdrant...")

try:
    # Try to retrieve the point
    points = qdrant_client.retrieve(
        collection_name="ecommerce",
        ids=[product_id]
    )
    
    if points:
        print(f"   ✅ Product found in Qdrant!")
        point = points[0]
        print(f"   Payload keys: {list(point.payload.keys())}")
        
        # Check what data is present
        has_image_data = any(key in point.payload for key in ['product_type', 'brand', 'colors', 'features'])
        has_text_data = any(key in point.payload for key in ['name', 'description', 'price'])
        
        print(f"   Has image features: {has_image_data}")
        print(f"   Has text data: {has_text_data}")
        
        if 'name' in point.payload:
            print(f"   Stored name: {point.payload['name']}")
        if 'price' in point.payload:
            print(f"   Stored price: ${point.payload['price']}")
        if 'product_type' in point.payload:
            print(f"   Stored product_type: {point.payload['product_type']}")
    else:
        print(f"   ⚠️ Product {product_id} NOT found in Qdrant")
        
except Exception as e:
    print(f"   ❌ Error retrieving: {e}")

# 5. Try to UPSERT text data for this ONE product
print(f"\n5. Attempting to add/update text data for product {product_id}...")

# Initialize embedder
print("   Loading sentence transformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create simple text embedding
product = result.data[0]
text_content = f"{product['name']} {product['description']} Price: ${product['price']}"
print(f"   Text content: {text_content[:100]}...")

embedding = embedder.encode(text_content).tolist()
print(f"   Embedding created (length: {len(embedding)})")

# Create point
from qdrant_client.models import PointStruct

point = PointStruct(
    id=product_id,
    vector=embedding,
    payload={
        "product_id": product_id,
        "name": product['name'],
        "description": product['description'][:500],
        "price": float(product['price']),
        "stock": product.get('stock', 0),
        "rating": float(product.get('avg_rating', 0)),
        "text_index_test": True,
        "test_timestamp": str(os.path.getctime(__file__))
    }
)

# Upsert to Qdrant
try:
    qdrant_client.upsert(
        collection_name="ecommerce",
        points=[point]
    )
    print(f"   ✅ Successfully UPSERTED product {product_id}")
    
    # Verify it worked
    verify = qdrant_client.retrieve(
        collection_name="ecommerce",
        ids=[product_id]
    )
    
    if verify:
        print(f"   ✅ Verification: Product now has {len(verify[0].payload.keys())} payload fields")
        print(f"   New payload keys: {list(verify[0].payload.keys())}")
        
        # Check if image data is still there
        if 'product_type' in verify[0].payload:
            print(f"   ✅ Image data preserved: product_type = {verify[0].payload['product_type']}")
        else:
            print(f"   ⚠️ WARNING: Image data appears to be LOST!")
            
except Exception as e:
    print(f"   ❌ Upsert failed: {e}")

# 6. Test search
print(f"\n6. Testing search after update...")
test_query = product['name'].split()[0]  # First word of product name
query_embedding = embedder.encode(test_query).tolist()

results = qdrant_client.search(
    collection_name="ecommerce",
    query_vector=query_embedding,
    limit=2
)

if results:
    print(f"   ✅ Search found {len(results)} results")
    for r in results:
        print(f"   - {r.payload.get('name', 'Unknown')} (score: {r.score:.3f})")
else:
    print(f"   ⚠️ No search results")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)