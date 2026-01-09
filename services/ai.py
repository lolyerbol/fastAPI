import io
import json
import uuid
from fastapi import UploadFile
from google import genai
from services.database import engine, ai_key
from services.tables import gemini_analysis_results
from PIL import Image
        

# Read and convert images
def read_images(image1_bytes: UploadFile,screenshot1_filename: str, image2_bytes: UploadFile,screenshot2_filename: str) -> dict:        

        
        image1 = Image.open(io.BytesIO(image1_bytes))
        image2 = Image.open(io.BytesIO(image2_bytes))
        
        # Initialize Gemini client and model
        client = genai.Client(api_key=ai_key)
        
        # Create structured prompt for analysis
        prompt = """Analyze these two screenshots and provide analysis in a structured JSON format with exactly these 4 fields:
        {
            "screenshot1_analysis": "Detailed analysis of the first screenshot",
            "screenshot2_analysis": "Detailed analysis of the second screenshot",
            "comparison_analysis": "Detailed comparison between the two screenshots highlighting differences and similarities",
            "future_perspectives": "Based on the data shown, what are the future perspectives or trends you can identify?"
        }
        Return ONLY valid JSON, no additional text."""
        
        # Generate response with images
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image1, image2]
        )
        
        # Parse the JSON response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        analysis_data = json.loads(response_text)
        
        # Store in database with structured fields
        insert_stmt = gemini_analysis_results.insert().values(
            id=uuid.uuid4(),
            screenshot1_filename=screenshot1_filename,
            screenshot2_filename=screenshot2_filename,

            screenshot1_analysis=analysis_data.get("screenshot1_analysis"),
            screenshot2_analysis=analysis_data.get("screenshot2_analysis"),
            comparison_analysis=analysis_data.get("comparison_analysis"),
            future_perspectives=analysis_data.get("future_perspectives")
        )
        with engine.begin() as conn:
            conn.execute(insert_stmt)
        
        return {
            "status": "success", 
            "analysis": analysis_data,
            "message": "Structured analysis completed and stored in database"
        }