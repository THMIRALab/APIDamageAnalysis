import io
import os
import sys
import io
import httpx
import tempfile

#from src.compat_hub import *
from src import utils_pipeline
from src.utils_pipeline import InputType, restore_ch_object
from text_to_image.run_text_to_image import main as text_to_image_main
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import Response
from typing import Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI()
#MODEL_GENERATOR_API_URL = "http://127.0.0.1:8001/generate_3d_model/"
MODEL_GENERATOR_API_URL = os.getenv("GENERATOR_API_URL", "http://dasch-engima-generator:8001/generate_3d_model/")
        

async def call_3d_model_generator(file_to_call, format_3d="glb"):
    url = f"{MODEL_GENERATOR_API_URL}?format={format_3d}"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            files=file_to_call,
            headers={"accept": "application/json"},
            timeout=600
        )
        
        if response.status_code == 200:
            '''
            file_extension = format_3d.lower()
            #filename = f"model.{file_extension}"
            content_disposition = response.headers.get("content-disposition", "")
            match = re.search(r'filename="?([^"]+)"?', content_disposition)
            filename = match.group(1) if match else f"model.{file_extension}"
            '''
            
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "application/octet-stream"),
                headers={
                    #"Content-Disposition": f"attachment; filename={filename}"
                    "Content-Disposition": response.headers.get("content-disposition", "attachment")
                }
            )
        else:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")


@app.post("/ch_object_generation/")
async def ch_object_generation(file: UploadFile = File(..., description="PNG image file of the CH object to generate the 3D Model."),
                               format_3d: str = "glb"):
    contents = await file.read()

    try:
        ch_object_image = Image.open(io.BytesIO(contents))
        ch_object_image = ch_object_image.convert("RGB")

        _, _, _, inpainted_object, _ = restore_ch_object(ch_object_image)

        img_byte_arr = io.BytesIO()
        inpainted_object.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        file_to_call = {"file": ("inpainted_object.png", img_byte_arr, "image/png")}
        return await call_3d_model_generator(file_to_call, format_3d=format_3d)
        '''
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )
        '''
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with 3D model service: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    '''img_byte_arr = io.BytesIO()
    final_damage_mask.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Reset position to beginning
    # Use the BytesIO object instead of the PIL image
    file_to_call = {"file": ("damage_mask.png", img_byte_arr, "image/png")}'''
    
    '''ch_object_shape_image = Image.fromarray(ch_object_shape)
    img_byte_arr = io.BytesIO()
    ch_object_shape_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Reset position to beginning
    file_to_call = {"file": ("shape_object.png", img_byte_arr, "image/png")}'''
    
    '''return Response(
        content=img_byte_arr.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=inpainted_object.png"}
    )'''


@app.post("/description_to_image/")
async def description_to_image(input_type: InputType = Form(...),
                               text: Optional[str] = Form(None), 
                               document: Union[UploadFile, str, None] = File(None),
                               audio: Union[UploadFile, str, None] = File(None),
                               format_3d: str = Form("glb")
                               ):
    """
    Generate an 2D-image from text, document, or audio input.
    - When input_type is 'text', provide text in the 'text' field
    - When input_type is 'document', upload a PDF or text file
    - When input_type is 'audio', upload an audio file
    """
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Validate inputs based on input_type
    if input_type == InputType.TEXT and not text:
        raise HTTPException(status_code=400, detail="Text input required when input_type is 'text'")
    
    if input_type == InputType.DOCUMENT and (not document or document == ''):
        raise HTTPException(status_code=400, detail="Document file required when input_type is 'document'")
    
    if input_type == InputType.AUDIO and (not audio or audio == ''):
        raise HTTPException(status_code=400, detail="Audio file required when input_type is 'audio'")
    
    try:
        temp_file_path = None
        
        if input_type == InputType.DOCUMENT and document:
            document_content = await document.read()

            if len(document_content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 10MB)")
            await document.seek(0)

            # Create temporary file
            suffix = os.path.splitext(document.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(document_content)
            
            generated_image = text_to_image_main(input_type, document_path=temp_file_path)
        
        elif input_type == InputType.AUDIO and audio:
            audio_content = await audio.read()

            if len(audio_content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large (max 10MB)")
            await audio.seek(0)

            # Create temporary file
            suffix = os.path.splitext(audio.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file_path = temp_file.name                
                temp_file.write(audio_content)
            
            generated_image = text_to_image_main(input_type, audio_path=temp_file_path)
        
        else:
            generated_image = text_to_image_main(input_type=InputType.TEXT, text=text)
        
        # Clean up temporary file if created
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        _, _, _, inpainted_object, _ = restore_ch_object(generated_image)

        img_byte_arr = io.BytesIO()
        inpainted_object.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        file_to_call = {"file": ("inpainted_object.png", img_byte_arr, "image/png")}
        return await call_3d_model_generator(file_to_call, format_3d=format_3d)
        '''
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )
        '''
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with 3D model service: {str(e)}")
    except Exception as e:
        # Clean up temporary file in case of errors
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
