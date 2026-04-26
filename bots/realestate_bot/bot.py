# bots/realestate_bot/bot.py
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealEstateBot:
    def __init__(self):
        self.bot_id = "realestate"
        self.name = "HomeMate"
        self.description = "AI assistant for real estate and property inquiries"
        
        # Initialize Groq client (no database needed)
        api_key = os.getenv("RE_GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
        if not api_key:
            print("⚠️ Warning: RE_GROQ_API_KEY not set")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=api_key)
        
        # Load all property and real estate documentation
        self.documents = self.load_document_context()
        
        print(f"✅ {self.name} initialized (documentation-only mode)")
    
    def load_document_context(self) -> Dict[str, str]:
        """Load static documentation about real estate"""
        return {
            "company_info": """
                HomeMate Real Estate is a premier property consultancy serving clients since 2015.
                We specialize in residential and commercial properties across major cities including 
                Karachi, Lahore, and Islamabad. Our mission is to help people find their dream homes 
                and investment properties.
            """,
            
            "property_listings": """
                CURRENT PROPERTY LISTINGS:
                
                RESIDENTIAL FOR SALE:
                1. 5 Bedrooms House in DHA Phase 5, Lahore
                - Price: PKR 25,000,000
                - Size: 3000 sq ft
                - 5 bedrooms, 6 bathrooms
                - Spacious house with modern amenities
                - Contact: John Doe (555-1234)
                
                2. 4 Bedrooms House in Bahria Enclave, Islamabad
                - Price: PKR 18,000,000
                - Size: 2000 sq ft
                - 4 bedrooms, 3 bathrooms
                - Beautiful house with modern design
                
                3. 5 Bedrooms House in E-11, Islamabad
                - Price: Contact for price
                - Size: 1700 sq ft
                - 5 bedrooms
                - Modern house with luxurious amenities
                
                RESIDENTIAL FOR RENT:
                4. 3 Bedrooms Apartment in F-8, Islamabad
                - Rent: PKR 50,000/month
                - Size: 2500 sq ft
                - 3 bedrooms, 2 bathrooms
                - Cozy apartment with great views
                
                5. 5 Bedrooms House in G-11, Islamabad
                - Rent: PKR 60,000/month
                - Size: 1800 sq ft
                - 5 bedrooms, 4 bathrooms
                - Spacious house with lush green surroundings
                
                COMMERCIAL FOR SALE:
                6. Commercial Building in Saddar, Karachi
                - Price: PKR 10,000,000
                - Size: 800 sq ft
                - Spacious commercial building for sale
                
                7. Office Space in Model Town, Lahore
                - Price: PKR 8,000,000
                - Size: 900 sq ft
                - Modern office space with great facilities
                
                COMMERCIAL FOR RENT:
                8. Office Space in Blue Area, Islamabad
                - Rent: PKR 150,000/month
                - Size: 5000 sq ft
                - Prime office space in the heart of Islamabad
                
                9. Shop in Liberty Market, Lahore
                - Rent: PKR 75,000/month
                - Size: 1000 sq ft
                - Prime location for your business
                
                10. Showroom in Johar Town, Lahore
                    - Rent: PKR 120,000/month
                    - Size: 1200 sq ft
                    - Spacious showroom for your business
                
                PLOTS FOR SALE:
                11. 10 Marla Plot in Bahria Town, Karachi
                    - Price: PKR 2,000,000
                    - Size: 10 marla
                    - Ideal residential plot for your dream home
                
                12. 15 Marla Plot in Gulshan-e-Iqbal, Karachi
                    - Price: PKR 3,000,000
                    - Size: 15 marla
                    - Residential plot in a well-established area
                
                13. 12 Marla Plot in Clifton, Karachi
                    - Price: PKR 2,400,000
                    - Size: 12 marla
                    - Perfect plot for residential development
                
                14. 14 Marla Plot in North Nazimabad, Karachi
                    - Price: PKR 3,200,000
                    - Size: 14 marla
                    - Ideal plot for investment or construction
                
                15. 8 Marla Plot in Malir, Karachi
                    - Price: PKR 1,600,000
                    - Size: 8 marla
                    - Residential plot in a developing area
                
                PLOTS FOR RENT:
                16. 10 Marla Plot in Gulistan-e-Jauhar, Karachi
                    - Rent: PKR 35,000/month
                    - Size: 10 marla
                    - Residential plot with easy access
            """,
            
            "property_types": """
                PROPERTY TYPES:
                
                Residential Properties:
                - Houses (various sizes: 1600-3000 sq ft)
                - Apartments (2500 sq ft)
                - Bedrooms range: 3-5 bedrooms
                
                Commercial Properties:
                - Office spaces (900-5000 sq ft)
                - Shops/Showrooms (1000-1200 sq ft)
                - Commercial buildings (800 sq ft)
                
                Plots:
                - Sizes: 8-15 marla
                - Locations: Bahria Town, Gulshan-e-Iqbal, Clifton, North Nazimabad, Malir, Gulistan-e-Jauhar
            """,
            
            "cities_areas": """
                CITIES AND AREAS:
                
                Karachi:
                - Bahria Town
                - Gulshan-e-Iqbal
                - Clifton
                - North Nazimabad
                - Malir
                - Gulistan-e-Jauhar
                - Saddar
                
                Lahore:
                - DHA Phase 5
                - Model Town
                - Johar Town
                - Liberty Market
                
                Islamabad:
                - Blue Area
                - F-8
                - Bahria Enclave
                - G-11
                - E-11
            """,
            
            "buying_process": """
                BUYING PROCESS:
                
                Step 1: Initial Consultation
                - Discuss requirements and budget
                - Review available properties
                
                Step 2: Property Search
                - Receive property recommendations
                - Schedule viewings (in-person or virtual)
                
                Step 3: Make an Offer
                - Submit offer through agent
                - Negotiate price and terms
                
                Step 4: Due Diligence
                - Property inspection
                - Title verification
                - Legal documentation review
                
                Step 5: Closing
                - Sign final documents
                - Transfer funds (PKR)
                - Receive keys
                
                Timeline: Typically 30-60 days from offer to closing
                Note: Prices are in Pakistani Rupees (PKR)
            """,
            
            "selling_process": """
                SELLING PROCESS:
                
                Step 1: Property Evaluation
                - Free property valuation
                - Market analysis
                - Pricing recommendation
                
                Step 2: Listing Preparation
                - Professional photography
                - Property description
                - Marketing materials
                
                Step 3: Marketing
                - List on property portals
                - Social media promotion
                - Open houses
                
                Step 4: Offer Management
                - Review offers
                - Negotiate terms
                - Accept offer
                
                Step 5: Closing
                - Handle paperwork
                - Transfer of ownership
                - Finalize sale
                
                Commission: Negotiable based on property value
            """,
            
            "rental_process": """
                RENTAL PROCESS:
                
                Step 1: Property Search
                - Browse available rentals
                - Schedule viewings
                
                Step 2: Application
                - Submit rental application
                - Provide references
                - Income verification
                
                Step 3: Agreement Signing
                - Review rental agreement terms
                - Sign contract
                - Pay security deposit
                
                Step 4: Move In
                - Property inspection
                - Get keys
                - Set up utilities
                
                Requirements:
                - Income proof
                - Security deposit: 2 months rent
                - Valid CNIC/ID
            """,
            
            "payment_guide": """
                PAYMENT INFORMATION:
                
                Accepted Payment Methods:
                - Bank Transfer
                - Online Banking
                - Cheque
                
                Currency:
                - All prices are in Pakistani Rupees (PKR)
                
                Typical Costs:
                - Down payment: 10-30% for purchases
                - Security deposit: 2 months rent for rentals
                - Agent commission: Negotiable
                - Legal fees: Varies by property
            """,
            
            "faqs": """
                FREQUENTLY ASKED QUESTIONS:
                
                Q: What currencies are prices in?
                A: All prices are in Pakistani Rupees (PKR).
                
                Q: Do you have properties in Islamabad?
                A: Yes, we have properties in Blue Area, F-8, Bahria Enclave, G-11, and E-11.
                
                Q: Do you have properties in Lahore?
                A: Yes, we have properties in DHA Phase 5, Model Town, Johar Town, and Liberty Market.
                
                Q: Do you have properties in Karachi?
                A: Yes, we have properties in Bahria Town, Gulshan-e-Iqbal, Clifton, North Nazimabad, Malir, Gulistan-e-Jauhar, and Saddar.
                
                Q: How can I contact the publisher?
                A: Each listing includes publisher name, phone number, and email.
                
                Q: Can I negotiate the price?
                A: Yes, most properties have room for negotiation.
                
                Q: Do you help with legal paperwork?
                A: Yes, we assist with all legal documentation.
                
                Q: Do you offer virtual tours?
                A: Yes, virtual tours can be arranged upon request.
            """,
            
            "contact_info": """
                CONTACT INFORMATION:
                
                Phone: +92 300 1234567
                Email: info@homemate.com
                WhatsApp: +92 300 1234567
                Office Hours: Mon-Fri 9am-7pm, Sat 10am-4pm
                
                Office Locations:
                - Main Office: DHA Phase 8, Karachi
                - Branch: Clifton Block 5, Karachi
                
                Social Media:
                - Facebook: /homemate
                - Instagram: @homemate_realestate
                - LinkedIn: /company/homemate
            """
        }
    def classify_intent(self, message: str) -> str:
        """Classify user intent based on keywords"""
        message_lower = message.lower()
        
        intents = {
            "buy_property": ["buy", "purchase", "investment", "buying", "cost", "price", "luxury villa", "apartment for sale"],
            "sell_property": ["sell", "listing", "selling", "valuation", "value", "worth"],
            "rent_property": ["rent", "lease", "rental", "tenant", "monthly rent"],
            "property_search": ["property", "home", "house", "villa", "apartment", "commercial", "office", "shop", "warehouse"],
            "financing": ["mortgage", "loan", "financing", "interest rate", "down payment", "fha", "va loan"],
            "investment": ["investment", "roi", "return", "cap rate", "cash flow", "flip"],
            "process": ["process", "steps", "how to", "procedure", "timeline"],
            "faq": ["faq", "question", "how", "what", "when", "where", "commission", "fee"],
            "contact": ["contact", "phone", "email", "office", "location", "hours"]
        }
        
        for intent, keywords in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        return "general"
    async def retrieve_context(self, message: str) -> tuple[str, List[str]]:
        """Retrieve relevant documentation based on intent"""
        context_parts = []
        context_sources = []
        intent = self.classify_intent(message)
        
        # Always include company info
        context_parts.append(self.documents["company_info"])
        context_sources.append("company_info")
        
        # Intent-specific documentation
        if intent in ["buy_property", "property_search", "general"]:
            context_parts.append(self.documents["property_listings"])
            context_sources.append("property_listings")
            context_parts.append(self.documents["buying_process"])
            context_sources.append("buying_process")
        
        if intent == "sell_property":
            context_parts.append(self.documents["selling_process"])
            context_sources.append("selling_process")
        
        if intent in ["rent_property", "property_search"]:
            # Check if rental_properties exists in documents
            if "rental_properties" in self.documents:
                context_parts.append(self.documents["rental_properties"])
                context_sources.append("rental_properties")
            context_parts.append(self.documents["rental_process"])
            context_sources.append("rental_process")
        
        if intent == "financing":
            if "financing_options" in self.documents:
                context_parts.append(self.documents["financing_options"])
                context_sources.append("financing_options")
            else:
                context_parts.append("We offer various financing options. Please contact our office for details.")
                context_sources.append("financing_info")
        
        if intent == "investment":
            # Check if investment_guide exists
            if "investment_guide" in self.documents:
                context_parts.append(self.documents["investment_guide"])
                context_sources.append("investment_guide")
            else:
                context_parts.append("We have investment properties available. Please contact our office for details.")
                context_sources.append("investment_info")
        
        if intent == "process":
            context_parts.append(self.documents["buying_process"])
            context_parts.append(self.documents["selling_process"])
            context_sources.append("process_info")
        
        if intent == "faq":
            context_parts.append(self.documents["faqs"])
            context_sources.append("faqs")
        
        if intent == "contact":
            context_parts.append(self.documents["contact_info"])
            context_sources.append("contact_info")
        
        # Add FAQ for general queries
        if intent == "general":
            context_parts.append(self.documents["faqs"])
            context_sources.append("faqs")
        
        # Add property types and cities for location-based queries
        if "property_types" in self.documents:
            context_parts.append(self.documents["property_types"])
            context_sources.append("property_types")
        
        if "cities_areas" in self.documents:
            context_parts.append(self.documents["cities_areas"])
            context_sources.append("cities_areas")
        
        return "\n\n---\n\n".join(context_parts), context_sources
    
    async def generate_response(self, message: str, context: str, history: List[Dict] = None) -> str:
        """Generate response using Groq"""
        if not self.groq_client:
            return "I apologize, but the AI service is not configured. Please contact support."
        
        system_prompt = f"""You are HomeMate, a friendly and professional real estate assistant.

DOCUMENTATION/CONTEXT:
{context}

YOUR ROLE:
- Help clients with property inquiries, buying, selling, and renting
- Provide accurate information from the documentation above
- Be warm, knowledgeable, and professional
- Keep responses concise (2-4 sentences)
- Never make up information not in context
- If asked about specific properties not listed, offer to connect with an agent
- Suggest similar properties when appropriate
- Always end with offering additional help

Remember: You only know about properties listed in the documentation. For specific inquiries about other properties, direct them to contact an agent.
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
            return "I'm having trouble right now. Please call our office for immediate assistance."
    
    async def chat(self, message: str, user_id: Optional[str] = None, 
                   conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Main chat method for the bot"""
        try:
            context, sources = await self.retrieve_context(message)
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
                "response": "I apologize, but I encountered an error. Please call our office or try again later.",
                "status": "error",
                "bot_name": self.name,
                "bot_id": self.bot_id,
                "timestamp": datetime.now().isoformat()
            }