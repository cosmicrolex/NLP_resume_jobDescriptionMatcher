import os
import sys
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Ensure the utils and app directories are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, '..', 'utils')
APP_DIR = BASE_DIR
sys.path.append(os.path.abspath(UTILS_DIR))
sys.path.append(os.path.abspath(APP_DIR))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 
#This means Python will now look in this upper-level directory when importing modules.
from backend.utils.pdf_parser import textextractionfunction
from tfidf_analyzer import analyze_resume_with_tfidf, analyze_job_description_with_tfidf, calculate_resume_job_similarity, comprehensive_resume_job_analysis

app = FastAPI()

OUTPUT_DIR = os.path.join(UTILS_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/analyze-resume/")
async def analyze_resume(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        output_path = os.path.join(OUTPUT_DIR, f"{file.filename}.txt")
        resume_text = textextractionfunction(file_path, output_path)
        tfidf_result = analyze_resume_with_tfidf(resume_text)
        try:
            os.remove(file_path)
            os.remove(output_path)
        except Exception:
            pass
        return JSONResponse(content={
            "extracted_text": resume_text,
            "tfidf_analysis": tfidf_result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

@app.post("/analyze-job-description/")
async def analyze_job_description(job_description: str = Form(...)):
    try:
        tfidf_result = analyze_job_description_with_tfidf(job_description)
        return JSONResponse(content={
            "job_description_text": job_description,
            "tfidf_analysis": tfidf_result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

@app.post("/match-resume-job/")
async def match_resume_job(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # Extract resume text
        file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        output_path = os.path.join(OUTPUT_DIR, f"{file.filename}.txt")
        resume_text = textextractionfunction(file_path, output_path)
        
        # Perform comprehensive analysis
        analysis_result = comprehensive_resume_job_analysis(resume_text, job_description)
        
        # Cleanup
        try:
            os.remove(file_path)
            os.remove(output_path)
        except Exception:
            pass
            
        return JSONResponse(content={
            "resume_text": resume_text,
            "job_description_text": job_description,
            "analysis": analysis_result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

@app.post("/analyze-job-description-pdf/")
async def analyze_job_description_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text from PDF
        output_path = os.path.join(OUTPUT_DIR, f"{file.filename}.txt")
        job_description_text = textextractionfunction(file_path, output_path)
        
        # Analyze with TF-IDF
        tfidf_result = analyze_job_description_with_tfidf(job_description_text)
        
        # Cleanup files
        try:
            os.remove(file_path)
            os.remove(output_path)
        except Exception:
            pass
        
        return JSONResponse(content={
            "extracted_text": job_description_text,
            "tfidf_analysis": tfidf_result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

@app.post("/match-resume-job-pdf/")
async def match_resume_job_pdf(
    file: UploadFile = File(...),
    jd_file: UploadFile = File(...)
):
    try:
        # Extract resume text
        resume_file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(resume_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        resume_output_path = os.path.join(OUTPUT_DIR, f"{file.filename}.txt")
        resume_text = textextractionfunction(resume_file_path, resume_output_path)
        
        # Extract job description text
        jd_file_path = os.path.join(OUTPUT_DIR, jd_file.filename)
        with open(jd_file_path, "wb") as f:
            content = await jd_file.read()
            f.write(content)
        jd_output_path = os.path.join(OUTPUT_DIR, f"{jd_file.filename}.txt")
        job_description_text = textextractionfunction(jd_file_path, jd_output_path)
        
        # Perform comprehensive analysis
        analysis_result = comprehensive_resume_job_analysis(resume_text, job_description_text)
        
        # Cleanup files
        try:
            os.remove(resume_file_path)
            os.remove(resume_output_path)
            os.remove(jd_file_path)
            os.remove(jd_output_path)
        except Exception:
            pass
            
        return JSONResponse(content={
            "resume_text": resume_text,
            "job_description_text": job_description_text,
            "analysis": analysis_result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

@app.get("/")
def home():
    return {"message": "Resume Analyzer API is running!"}
