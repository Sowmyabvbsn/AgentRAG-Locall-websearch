import os
import logging
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import librosa
import soundfile as sf
import whisper
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class MultiModalProcessor:
    """Handles processing of various input types: PDFs, images, audio, documents"""
    
    def __init__(self):
        self.whisper_model = None
        self.easyocr_reader = None
        self._init_models()
    
    def _init_models(self):
        """Initialize models for audio and OCR processing"""
        try:
            # Initialize Whisper for audio transcription
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded")
            
            # Initialize EasyOCR for text extraction
            self.easyocr_reader = easyocr.Reader(['en'])
            logger.info("✅ EasyOCR initialized")
            
        except Exception as e:
            logger.warning(f"Error initializing models: {e}")
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files and extract text"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update({
                    "source": os.path.basename(file_path),
                    "type": "pdf",
                    "processed_by": "PyPDFLoader"
                })
            
            logger.info(f"📄 Processed PDF: {len(documents)} pages")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def process_image(self, file_path: str) -> List[Document]:
        """Process images and extract text using OCR"""
        documents = []
        
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                logger.error(f"Could not load image: {file_path}")
                return []
            
            # Extract text using Tesseract
            tesseract_text = ""
            try:
                tesseract_text = pytesseract.image_to_string(image)
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
            
            # Extract text using EasyOCR
            easyocr_text = ""
            if self.easyocr_reader:
                try:
                    results = self.easyocr_reader.readtext(image)
                    easyocr_text = " ".join([result[1] for result in results])
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            # Combine OCR results
            combined_text = f"Tesseract OCR: {tesseract_text}\n\nEasyOCR: {easyocr_text}"
            
            if combined_text.strip():
                doc = Document(
                    page_content=combined_text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": "image",
                        "processed_by": "OCR",
                        "ocr_methods": ["tesseract", "easyocr"]
                    }
                )
                documents.append(doc)
                logger.info(f"🖼️ Processed image: extracted {len(combined_text)} characters")
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
        
        return documents
    
    def process_audio(self, file_path: str) -> List[Document]:
        """Process audio files and extract text using speech recognition"""
        documents = []
        
        try:
            if not self.whisper_model:
                logger.error("Whisper model not available")
                return []
            
            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(file_path)
            transcription = result["text"]
            
            if transcription.strip():
                # Get audio metadata
                try:
                    audio_data, sample_rate = librosa.load(file_path)
                    duration = len(audio_data) / sample_rate
                except:
                    duration = "unknown"
                
                doc = Document(
                    page_content=transcription,
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": "audio",
                        "processed_by": "Whisper",
                        "duration": duration,
                        "language": result.get("language", "unknown")
                    }
                )
                documents.append(doc)
                logger.info(f"🎵 Processed audio: {len(transcription)} characters transcribed")
            
        except Exception as e:
            logger.error(f"Error processing audio {file_path}: {e}")
        
        return documents
    
    def process_text_document(self, file_path: str) -> List[Document]:
        """Process text documents (txt, md, etc.)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "text",
                    "processed_by": "direct_read"
                }
            )
            
            logger.info(f"📝 Processed text document: {len(content)} characters")
            return [doc]
            
        except Exception as e:
            logger.error(f"Error processing text document {file_path}: {e}")
            return []
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process a file based on its extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.process_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            return self.process_image(file_path)
        elif file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            return self.process_audio(file_path)
        elif file_ext in ['.txt', '.md', '.rst']:
            return self.process_text_document(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []