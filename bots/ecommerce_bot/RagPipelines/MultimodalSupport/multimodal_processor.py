import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Union, Dict, Optional
import requests
from groq import Groq
from bots.ecommerce_bot.RagPipelines.Rewriting.QueryRewriter import QueryRewritingPipeline
class GroqOnlyMultimodalProcessor:
    """Use Groq for all multimodal processing with current models"""
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.text_model = "llama-3.1-8b-instant"
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
        if self.deepgram_api_key:
            transcript = await self._transcribe_with_deepgram(audio_data, language)
        else:
            return "Voice transcription requires Deepgram, AssemblyAI, or HuggingFace API key."
        
        if transcript and transcript != "":
            return await self._enhance_transcript_with_groq(transcript)
        return ""
    
    async def _enhance_transcript_with_groq(self, transcript: str) -> str:
        """Use Groq to clean up transcript"""
        if not transcript:
            return ""
        
        try:
            enhancement = self.groq_client.chat.completions.create(
                model=self.text_model,
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
            
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages[:10]:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
            
            full_text = "\n\n".join(text_content)
            
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
    "extracted_data": {}
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
            
            if 'temp_pdf' in locals() and Path(pdf_path).exists():
                os.unlink(pdf_path)
            
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

