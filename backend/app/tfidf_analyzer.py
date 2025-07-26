import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """Minimal preprocessing to preserve meaningful terms"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep letters, numbers, spaces, and hyphens
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
    
    # Remove standalone years (2020, 2021, etc.)
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)
    
    # Remove very short words (less than 2 chars) and very long words (more than 20 chars)
    words = text.split()
    words = [word for word in words if 2 <= len(word) <= 20]
    
    # Minimal stopword removal - only remove very common words
    minimal_stopwords = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'among', 'this', 'that', 'these', 'those', 'is', 'was', 'are',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
    }
    
    # Keep technical and meaningful terms
    filtered_words = [word for word in words if word not in minimal_stopwords]
    
    result = ' '.join(filtered_words)
    print(f"DEBUG - Preprocessed text preview: {result[:200]}...")  # Debug output
    return result

def analyze_resume_with_tfidf(resume_text):
    try:
        print(f"DEBUG - Original resume text length: {len(resume_text)}")
        processed_text = preprocess_text(resume_text)
        print(f"DEBUG - Processed resume text length: {len(processed_text)}")
        
        if not processed_text.strip():
            return {"error": "No valid text extracted for TF-IDF analysis"}

        vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,  # Changed from 0.9 to 1.0 for single documents
            stop_words=None,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        print(f"DEBUG - Resume features found: {len(feature_names)}")
        print(f"DEBUG - Top resume features: {feature_names[:10]}")
        
        keyword_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
        top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return {
            "top_keywords": [{"term": term, "score": round(score, 4)} for term, score in top_keywords if score > 0]
        }
    except Exception as e:
        print(f"DEBUG - Resume analysis error: {str(e)}")
        return {"error": f"TF-IDF analysis failed: {str(e)}"}

def analyze_job_description_with_tfidf(job_description_text):
    try:
        print(f"DEBUG - Original job desc text length: {len(job_description_text)}")
        processed_text = preprocess_text(job_description_text)
        print(f"DEBUG - Processed job desc text length: {len(processed_text)}")
        
        if not processed_text.strip():
            return {"error": "No valid text extracted for TF-IDF analysis"}

        vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,  # Changed from 0.9 to 1.0 for single documents
            stop_words=None,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        print(f"DEBUG - Job desc features found: {len(feature_names)}")
        print(f"DEBUG - Top job desc features: {feature_names[:10]}")
        
        keyword_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
        top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return {
            "top_keywords": [{"term": term, "score": round(score, 4)} for term, score in top_keywords if score > 0]
        }
    except Exception as e:
        print(f"DEBUG - Job desc analysis error: {str(e)}")
        return {"error": f"TF-IDF analysis failed: {str(e)}"}

def calculate_resume_job_similarity(resume_text, job_description_text):
    try:
        print("DEBUG - Starting similarity calculation...")
        processed_resume = preprocess_text(resume_text)
        processed_job_desc = preprocess_text(job_description_text)
        
        print(f"DEBUG - Processed resume length: {len(processed_resume)}")
        print(f"DEBUG - Processed job desc length: {len(processed_job_desc)}")
        
        if not processed_resume.strip() or not processed_job_desc.strip():
            return {"error": "One or both texts are empty after preprocessing"}

        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,  # Changed from 0.9 to 1.0 for two documents
            stop_words=None,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        # Combine texts for vectorization
        combined_texts = [processed_resume, processed_job_desc]
        tfidf_matrix = vectorizer.fit_transform(combined_texts)
        
        feature_names = vectorizer.get_feature_names_out()
        print(f"DEBUG - Total features in similarity: {len(feature_names)}")
        print(f"DEBUG - Sample features: {feature_names[:10]}")
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = similarity_matrix[0][0]
        
        print(f"DEBUG - Raw similarity score: {similarity_score}")
        
        # Get feature analysis
        resume_scores = tfidf_matrix[0].toarray()[0]
        job_desc_scores = tfidf_matrix[1].toarray()[0]
        
        # Find common terms with detailed debugging
        common_terms = []
        for i, feature in enumerate(feature_names):
            if resume_scores[i] > 0 and job_desc_scores[i] > 0:
                common_terms.append({
                    "term": feature,
                    "resume_score": round(resume_scores[i], 4),
                    "job_desc_score": round(job_desc_scores[i], 4),
                    "combined_importance": round((resume_scores[i] + job_desc_scores[i]) / 2, 4)
                })
                print(f"DEBUG - Common term found: {feature} (resume: {resume_scores[i]:.4f}, job: {job_desc_scores[i]:.4f})")
        
        print(f"DEBUG - Common terms found: {len(common_terms)}")
        
        common_terms = sorted(common_terms, key=lambda x: x["combined_importance"], reverse=True)[:15]
        
        # Match quality
        if similarity_score >= 0.3:
            match_quality = "Excellent Match"
        elif similarity_score >= 0.2:
            match_quality = "Good Match"
        elif similarity_score >= 0.1:
            match_quality = "Fair Match"
        else:
            match_quality = "Poor Match"
        
        return {
            "similarity_score": round(similarity_score, 4),
            "match_quality": match_quality,
            "common_keywords": common_terms,
            "total_features": len(feature_names)
        }
        
    except Exception as e:
        print(f"DEBUG - Similarity calculation error: {str(e)}")
        return {"error": f"Similarity calculation failed: {str(e)}"}

def comprehensive_resume_job_analysis(resume_text, job_description_text):
    try:
        print("DEBUG - Starting comprehensive analysis...")
        resume_analysis = analyze_resume_with_tfidf(resume_text)
        job_desc_analysis = analyze_job_description_with_tfidf(job_description_text)
        similarity_analysis = calculate_resume_job_similarity(resume_text, job_description_text)
        
        return {
            "resume_analysis": resume_analysis,
            "job_description_analysis": job_desc_analysis,
            "similarity_analysis": similarity_analysis
        }
        
    except Exception as e:
        print(f"DEBUG - Comprehensive analysis error: {str(e)}")
        return {"error": f"Comprehensive analysis failed: {str(e)}"}
