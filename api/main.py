"""FastAPI server for SpanBERT-CRF model inference."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import torch

app = FastAPI(
    title="SpanBERT-CRF API",
    description="API for Question Answering and Named Entity Recognition using SpanBERT with CRF",
    version="1.0.0"
)

# Global inference objects
qa_model = None
ner_model = None


class QARequest(BaseModel):
    """Request model for QA task."""
    context: str
    question: str


class NERRequest(BaseModel):
    """Request model for NER task."""
    text: str


class QAResponse(BaseModel):
    """Response model for QA task."""
    answer: str
    confidence: float
    start_token: int
    end_token: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class NEREntity(BaseModel):
    """Entity in NER response."""
    type: str
    text: str
    start_token: int
    end_token: int


class NERResponse(BaseModel):
    """Response model for NER task."""
    entities: List[NEREntity]


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global qa_model, ner_model
    
    try:
        from src.inference import load_inference_model
        
        # Load QA model
        try:
            qa_model = load_inference_model(
                'models/spanbert_qa_crf.pt',
                task_type='qa',
                use_crf=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("QA model loaded successfully")
        except FileNotFoundError:
            print("QA model not found, will load on first request")
        
        # Load NER model
        try:
            ner_model = load_inference_model(
                'models/spanbert_ner_crf.pt',
                task_type='ner',
                use_crf=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("NER model loaded successfully")
        except FileNotFoundError:
            print("NER model not found, will load on first request")
    
    except Exception as e:
        print(f"Warning: Could not preload models: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SpanBERT-CRF API",
        "endpoints": {
            "/qa": "POST - Question Answering",
            "/ner": "POST - Named Entity Recognition",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/qa", response_model=QAResponse)
async def question_answering(request: QARequest):
    """
    Answer a question based on the provided context.
    
    Args:
        request: QARequest with context and question
    
    Returns:
        QAResponse with predicted answer and metadata
    """
    global qa_model
    
    if qa_model is None:
        from src.inference import load_inference_model
        qa_model = load_inference_model(
            'models/spanbert_qa_crf.pt',
            task_type='qa',
            use_crf=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    try:
        result = qa_model.predict(request.context, request.question)
        
        return QAResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            start_token=result['start_token'],
            end_token=result['end_token'],
            start_char=result.get('start_char'),
            end_char=result.get('end_char')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ner", response_model=NERResponse)
async def named_entity_recognition(request: NERRequest):
    """
    Extract named entities from text.
    
    Args:
        request: NERRequest with text
    
    Returns:
        NERResponse with list of entities
    """
    global ner_model
    
    if ner_model is None:
        from src.inference import load_inference_model
        ner_model = load_inference_model(
            'models/spanbert_ner_crf.pt',
            task_type='ner',
            use_crf=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    try:
        entities = ner_model.predict(request.text)
        
        return NERResponse(
            entities=[
                NEREntity(
                    type=e['type'],
                    text=e['text'],
                    start_token=e['start_token'],
                    end_token=e['end_token']
                )
                for e in entities
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa/batch")
async def question_answering_batch(requests: List[QARequest]):
    """Process multiple QA requests in batch."""
    global qa_model
    
    if qa_model is None:
        from src.inference import load_inference_model
        qa_model = load_inference_model(
            'models/spanbert_qa_crf.pt',
            task_type='qa',
            use_crf=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    contexts = [r.context for r in requests]
    questions = [r.question for r in requests]
    
    results = qa_model.predict_batch(contexts, questions)
    
    return [
        QAResponse(
            answer=r['answer'],
            confidence=r['confidence'],
            start_token=r['start_token'],
            end_token=r['end_token'],
            start_char=r.get('start_char'),
            end_char=r.get('end_char')
        )
        for r in results
    ]


@app.post("/ner/batch")
async def named_entity_recognition_batch(requests: List[NERRequest]):
    """Process multiple NER requests in batch."""
    global ner_model
    
    if ner_model is None:
        from src.inference import load_inference_model
        ner_model = load_inference_model(
            'models/spanbert_ner_crf.pt',
            task_type='ner',
            use_crf=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    texts = [r.text for r in requests]
    results = ner_model.predict_batch(texts)
    
    return [
        NERResponse(
            entities=[
                NEREntity(
                    type=e['type'],
                    text=e['text'],
                    start_token=e['start_token'],
                    end_token=e['end_token']
                )
                for e in entities
            ]
        )
        for entities in results
    ]


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
