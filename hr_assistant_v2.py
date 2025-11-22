"""
THE HR - HR Intelligent Assistant
A comprehensive AI-powered HR policy management system with employee Q&A and admin document management.
Enhanced version with improved document analysis and conversational AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import io
import base64
import re
from typing import List, Dict, Any, Optional, Tuple
import time
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
import tempfile
import os
import pickle
from dotenv import load_dotenv

# Load environment variables from .env (for local testing)
load_dotenv()

# Try to import PyPDF2, but provide fallback if not available
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PyPDF2 not installed. PDF upload functionality limited. Run: pip install PyPDF2")

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Admin credentials (for demo)
ADMIN_USERNAME = "hradmin"
ADMIN_PASSWORD = "hrpass123"

# File paths for persistence
DOCUMENTS_FILE = "hr_documents.json"
CHAT_HISTORY_FILE = "chat_history.json"

# Page configuration - Force light theme
st.set_page_config(
    page_title="THE HR - HR Intelligent Assistant",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for light mode UI with better visibility
st.markdown("""
<style>
    /* Force light theme background */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Main content area */
    .main {
        background-color: #ffffff !important;
    }
    
    /* All text should be dark for visibility */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #1e1e1e !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px 10px 0 0;
        color: #1e1e1e !important;
    }
    
    /* Ensure tab labels and any nested elements are dark when NOT selected (override global white rule) */
    .stTabs [data-baseweb="tab"] *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: #1e1e1e !important;
    }

    /* Keep selected tab styling explicit */
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border-color: #4CAF50 !important;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        border: 1px solid #dee2e6;
    }
    
    .user-message {
        background-color: #E8F5E9;
        align-self: flex-end;
        color: #1e1e1e !important;
        max-width: 70%;
    }
    
    .assistant-message {
        background-color: #F5F5F5;
        align-self: flex-start;
        color: #1e1e1e !important;
        max-width: 70%;
    }
    
    /* Policy health cards */
    .policy-health-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #1e1e1e !important;
    }
    
    .severity-high {
        border-left: 4px solid #ff4444;
        background-color: #ffebee;
    }
    
    .severity-medium {
        border-left: 4px solid #ffaa00;
        background-color: #fff8e1;
    }
    
    .severity-low {
        border-left: 4px solid #00C851;
        background-color: #e8f5e9;
    }
    
    /* Document cards */
    .document-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1e1e1e !important;
    }
    
    .document-card b {
        color: #1e1e1e !important;
    }
    
    .document-card small {
        color: #6c757d !important;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-card h2, .stat-card h3, .stat-card p {
        color: white !important;
    }
    
    /* Headers and titles */
    .main-header {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1, .main-header p {
        color: white !important;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #f0f9ff;
        border: 1px solid #0284c7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Answer highlight */
    .answer-highlight {
        background-color: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Button styling - Ensuring white text on all buttons */
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        font-weight: 500 !important;
    }
    
    .stButton > button:hover {
        background-color: #45a049 !important;
        color: white !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background-color: #6c757d !important;
        color: white !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #5a6268 !important;
        color: white !important;
    }
    
    /* Form submit buttons */
    button[type="submit"], 
    .stForm button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background-color: #007bff !important;
        color: white !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #0056b3 !important;
        color: white !important;
    }
    
    /* File uploader button */
    .stFileUploader > div > button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Ensure all text inside buttons is white */
    .stButton > button p, 
    .stButton > button span,
    .stButton > button div,
    button p,
    button span {
        color: white !important;
    }
    
    /* Fix for any button with dark background */
    button {
        color: white !important;
    }
    
    [data-testid="baseButton-secondary"] {
        background-color: #6c757d !important;
        color: white !important;
    }
    
    [data-testid="baseButton-primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Document and Policy Classes
@dataclass
class Document:
    id: str
    name: str
    content: str
    type: str
    upload_date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    
    def to_dict(self):
        """Convert document to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'content': self.content,
            'type': self.type,
            'upload_date': self.upload_date.isoformat(),
            'metadata': self.metadata,
            'embeddings': self.embeddings.tolist() if isinstance(self.embeddings, np.ndarray) else self.embeddings
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create document from dictionary"""
        doc = cls(
            id=data['id'],
            name=data['name'],
            content=data['content'],
            type=data['type'],
            upload_date=datetime.fromisoformat(data['upload_date']),
            metadata=data.get('metadata', {}),
            embeddings=np.array(data['embeddings']) if data.get('embeddings') else None
        )
        return doc

@dataclass
class PolicyIssue:
    severity: str
    type: str
    description: str
    source_docs: List[str]
    suggested_fix: str
    location: Optional[str] = None

# Persistence Functions
def save_documents():
    """Save documents to file for persistence"""
    try:
        if st.session_state.documents:
            docs_dict = {
                doc_id: doc.to_dict() 
                for doc_id, doc in st.session_state.documents.items()
            }
            with open(DOCUMENTS_FILE, 'w') as f:
                json.dump(docs_dict, f)
    except Exception as e:
        st.error(f"Error saving documents: {str(e)}")

def load_documents():
    """Load documents from file"""
    try:
        if os.path.exists(DOCUMENTS_FILE):
            with open(DOCUMENTS_FILE, 'r') as f:
                docs_dict = json.load(f)
            st.session_state.documents = {
                doc_id: Document.from_dict(doc_data)
                for doc_id, doc_data in docs_dict.items()
            }
            return True
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
    return False

def save_chat_history():
    """Save chat history to file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(st.session_state.chat_history, f)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                st.session_state.chat_history = json.load(f)
            return True
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
    return False

# Initialize session state with persistence
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.authenticated = False
    st.session_state.documents = {}
    st.session_state.chat_history = []
    st.session_state.document_embeddings = {}
    st.session_state.sample_docs_loaded = False
    
    # Load persisted data
    load_documents()
    load_chat_history()

class PolicyAnalyzer:
    """Analyzes HR policies for contradictions, missing clauses, and ambiguities"""
    
    @staticmethod
    def analyze_document(doc: Document, all_docs: Dict[str, Document]) -> List[PolicyIssue]:
        """Analyze a document for policy issues"""
        issues = []
        content_lower = doc.content.lower()
        
        # Check for common missing clauses
        required_sections = [
            ("leave policy", "annual leave", "vacation"),
            ("code of conduct", "behavior", "ethics"),
            ("grievance", "complaint", "dispute resolution"),
            ("termination", "resignation", "exit"),
            ("benefits", "compensation", "salary"),
            ("remote work", "work from home", "hybrid"),
            ("data protection", "privacy", "confidential"),
            ("health and safety", "workplace safety", "emergency")
        ]
        
        for section_variants in required_sections:
            if not any(term in content_lower for term in section_variants):
                issues.append(PolicyIssue(
                    severity="medium",
                    type="missing_clause",
                    description=f"Missing section: {section_variants[0].title()}",
                    source_docs=[doc.name],
                    suggested_fix=f"Add a comprehensive {section_variants[0]} section covering key aspects and procedures.",
                    location=None
                ))
        
        # Check for contradictions with other documents
        for other_id, other_doc in all_docs.items():
            if other_id != doc.id:
                contradictions = PolicyAnalyzer._find_contradictions(doc, other_doc)
                issues.extend(contradictions)
        
        # Check for ambiguous language
        ambiguous_phrases = [
            ("may be", "Consider using 'will be' or 'shall be' for clarity"),
            ("should", "Consider using 'must' or 'shall' for mandatory requirements"),
            ("reasonable", "Define specific criteria or timeframes"),
            ("as soon as possible", "Specify exact timeframe (e.g., within 48 hours)"),
            ("appropriate", "Provide specific guidelines or examples"),
        ]
        
        for phrase, suggestion in ambiguous_phrases:
            if phrase in content_lower:
                issues.append(PolicyIssue(
                    severity="low",
                    type="ambiguity",
                    description=f"Ambiguous language detected: '{phrase}'",
                    source_docs=[doc.name],
                    suggested_fix=suggestion,
                    location=f"Found in document: {doc.name}"
                ))
        
        return issues
    
    @staticmethod
    def _find_contradictions(doc1: Document, doc2: Document) -> List[PolicyIssue]:
        """Find potential contradictions between two documents"""
        contradictions = []
        
        patterns = [
            (r"annual leave[:\s]+(\d+)\s*days", "Annual leave entitlement"),
            (r"notice period[:\s]+(\d+)\s*(days|weeks|months)", "Notice period requirement"),
            (r"probation[:\s]+(\d+)\s*months", "Probation period"),
            (r"working hours[:\s]+(\d+)\s*hours", "Working hours"),
        ]
        
        for pattern, description in patterns:
            matches1 = re.findall(pattern, doc1.content.lower())
            matches2 = re.findall(pattern, doc2.content.lower())
            
            if matches1 and matches2 and matches1[0] != matches2[0]:
                contradictions.append(PolicyIssue(
                    severity="high",
                    type="contradiction",
                    description=f"Conflicting {description} between documents",
                    source_docs=[doc1.name, doc2.name],
                    suggested_fix=f"Reconcile {description} across documents. {doc1.name} states '{matches1[0]}' while {doc2.name} states '{matches2[0]}'",
                    location=f"Found in {doc1.name} and {doc2.name}"
                ))
        
        return contradictions

class DocumentProcessor:
    """Handles document upload, parsing, and indexing"""
    
    @staticmethod
    def process_pdf(file_content: bytes, filename: str) -> Document:
        """Process PDF file and extract text"""
        try:
            if PDF_SUPPORT:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                doc_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:8]
                
                return Document(
                    id=doc_id,
                    name=filename,
                    content=text,
                    type="pdf",
                    upload_date=datetime.now(),
                    metadata={
                        "pages": len(pdf_reader.pages),
                        "file_size": len(file_content)
                    }
                )
            else:
                st.warning("PDF processing library not available. Using fallback text extraction.")
                doc_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:8]
                return Document(
                    id=doc_id,
                    name=filename,
                    content=f"PDF Document: {filename}\n[PDF content would be extracted here with PyPDF2 installed]",
                    type="pdf",
                    upload_date=datetime.now(),
                    metadata={
                        "file_size": len(file_content),
                        "note": "Install PyPDF2 for full PDF processing"
                    }
                )
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    
    @staticmethod
    def process_text(file_content: bytes, filename: str) -> Document:
        """Process text file"""
        try:
            text = file_content.decode('utf-8')
            doc_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:8]
            
            return Document(
                id=doc_id,
                name=filename,
                content=text,
                type="txt",
                upload_date=datetime.now(),
                metadata={
                    "file_size": len(file_content)
                }
            )
        except Exception as e:
            st.error(f"Error processing text file: {str(e)}")
            return None
    
    @staticmethod
    def create_embeddings(text: str) -> np.ndarray:
        """Create simple embeddings for text (simulated)"""
        words = text.lower().split()[:100]
        embedding = np.array([hash(word) % 1000 for word in words])
        if len(embedding) < 100:
            embedding = np.pad(embedding, (0, 100 - len(embedding)))
        else:
            embedding = embedding[:100]
        return embedding / (np.linalg.norm(embedding) + 1e-8)

class HRAssistant:
    """Enhanced AI assistant for answering HR questions with better document understanding"""
    
    def __init__(self):
        self.groq_api_key = GROQ_API_KEY
        
    def answer_question(self, question: str, documents: Dict[str, Document]) -> Tuple[str, List[str]]:
        """Answer a question based on available documents with improved analysis"""
        if not documents:
            return "No HR documents are currently loaded. Please ask your HR administrator to upload relevant policies.", []
        
        # Find relevant documents
        relevant_docs = self._retrieve_relevant_documents(question, documents)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the available HR documents. Please contact your HR department for assistance.", []
        
        # Generate comprehensive answer based on documents
        answer = self._generate_comprehensive_answer(question, relevant_docs)
        doc_citations = [doc.name for doc in relevant_docs]
        
        return answer, doc_citations
    
    def _retrieve_relevant_documents(self, question: str, documents: Dict[str, Document], top_k: int = 3) -> List[Document]:
        """Retrieve most relevant documents for the question with improved scoring"""
        question_lower = question.lower()
        scored_docs = []
        
        # Define keyword weights for different topics
        topic_keywords = {
            'leave': ['leave', 'vacation', 'holiday', 'time off', 'pto', 'sick', 'annual', 'maternity', 'paternity'],
            'benefits': ['benefit', 'insurance', 'health', 'medical', 'dental', 'vision', '401k', 'retirement', 'pension'],
            'salary': ['salary', 'compensation', 'pay', 'wage', 'bonus', 'increment', 'raise', 'payment'],
            'conduct': ['conduct', 'behavior', 'ethics', 'discipline', 'policy', 'rule', 'guideline'],
            'remote': ['remote', 'work from home', 'wfh', 'hybrid', 'flexible', 'telecommute'],
            'performance': ['performance', 'review', 'appraisal', 'evaluation', 'feedback', 'goal', 'kpi'],
            'termination': ['termination', 'resignation', 'notice', 'exit', 'separation', 'layoff'],
            'training': ['training', 'development', 'learning', 'skill', 'education', 'course', 'certification']
        }
        
        for doc_id, doc in documents.items():
            doc_content_lower = doc.content.lower()
            score = 0
            
            # Direct keyword matching
            keywords = question_lower.split()
            for keyword in keywords:
                if len(keyword) > 2:  # Skip very short words
                    occurrences = doc_content_lower.count(keyword)
                    score += occurrences * 2
            
            # Topic-based scoring
            for topic, topic_words in topic_keywords.items():
                for word in topic_words:
                    if word in question_lower:
                        # Question contains topic word, check document
                        for topic_word in topic_words:
                            if topic_word in doc_content_lower:
                                score += 3
            
            # Bonus for exact phrase matching
            if len(question) > 10:
                phrases = [question_lower[i:i+20] for i in range(0, len(question_lower)-20, 10)]
                for phrase in phrases:
                    if phrase in doc_content_lower:
                        score += 10
            
            # Title relevance bonus
            if any(keyword in doc.name.lower() for keyword in keywords if len(keyword) > 3):
                score += 5
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
    
    def _extract_relevant_sections(self, doc: Document, question: str, max_chars: int = 500) -> str:
        """Extract the most relevant sections from a document"""
        question_lower = question.lower()
        keywords = [word for word in question_lower.split() if len(word) > 3]
        
        # Split document into paragraphs
        paragraphs = doc.content.split('\n')
        scored_paragraphs = []
        
        for para in paragraphs:
            if len(para.strip()) < 20:  # Skip very short paragraphs
                continue
            para_lower = para.lower()
            score = sum(1 for keyword in keywords if keyword in para_lower)
            if score > 0:
                scored_paragraphs.append((score, para))
        
        # Sort by relevance and take top paragraphs
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        
        relevant_text = ""
        for _, para in scored_paragraphs[:3]:  # Take top 3 paragraphs
            if len(relevant_text) + len(para) < max_chars:
                relevant_text += para + "\n\n"
        
        return relevant_text.strip() if relevant_text else doc.content[:max_chars]
    
    def _generate_comprehensive_answer(self, question: str, relevant_docs: List[Document]) -> str:
        """Generate a comprehensive answer with specific information extraction"""
        question_lower = question.lower()
        
        # Extract relevant sections from each document
        all_relevant_info = []
        for doc in relevant_docs:
            relevant_section = self._extract_relevant_sections(doc, question)
            if relevant_section:
                all_relevant_info.append({
                    'doc_name': doc.name,
                    'content': relevant_section
                })
        
        # Process specific question types with enhanced extraction
        if "how many" in question_lower or "number of" in question_lower:
            return self._answer_quantitative_question(question_lower, all_relevant_info, relevant_docs)
        
        elif "what is" in question_lower or "what are" in question_lower:
            return self._answer_definition_question(question_lower, all_relevant_info, relevant_docs)
        
        elif "how to" in question_lower or "how do" in question_lower or "process" in question_lower:
            return self._answer_procedural_question(question_lower, all_relevant_info, relevant_docs)
        
        elif "when" in question_lower:
            return self._answer_temporal_question(question_lower, all_relevant_info, relevant_docs)
        
        elif "who" in question_lower:
            return self._answer_person_question(question_lower, all_relevant_info, relevant_docs)
        
        elif any(word in question_lower for word in ['policy', 'rule', 'guideline', 'requirement']):
            return self._answer_policy_question(question_lower, all_relevant_info, relevant_docs)
        
        else:
            # General answer with information synthesis
            return self._synthesize_general_answer(question_lower, all_relevant_info, relevant_docs)
    
    def _answer_quantitative_question(self, question: str, info: List[Dict], docs: List[Document]) -> str:
        """Answer questions about quantities, numbers, amounts"""
        numbers_found = []
        
        for doc in docs:
            content = doc.content.lower()
            
            # Look for leave days
            if "leave" in question or "vacation" in question or "holiday" in question:
                patterns = [
                    r"annual leave[:\s]+(\d+)\s*days",
                    r"vacation[:\s]+(\d+)\s*days",
                    r"sick leave[:\s]+(\d+)\s*days",
                    r"(\d+)\s*days?\s*(?:of\s*)?(?:annual\s*)?leave",
                    r"personal leave[:\s]+(\d+)\s*days",
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        for match in matches:
                            numbers_found.append(f"According to {doc.name}: {match} days")
            
            # Look for other numerical information
            elif "hour" in question:
                patterns = [r"(\d+)\s*hours?\s*(?:per\s*)?week", r"working hours[:\s]+(\d+)"]
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        numbers_found.append(f"According to {doc.name}: {matches[0]} hours")
            
            elif "salary" in question or "bonus" in question or "pay" in question:
                patterns = [
                    r"(\d+[%])\s*(?:of\s*)?(?:base\s*)?salary",
                    r"bonus[:\s]+(?:up\s*to\s*)?(\d+[%])",
                    r"\$(\d+(?:,\d+)*)"
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        numbers_found.append(f"According to {doc.name}: {matches[0]}")
        
        if numbers_found:
            return f"Based on the HR documents:\n" + "\n".join(f"• {item}" for item in numbers_found[:3])
        else:
            return f"I couldn't find specific numerical information about your query in the documents. The documents available are: {', '.join(doc.name for doc in docs)}. Please check these documents directly or contact HR for specific numbers."
    
    def _answer_definition_question(self, question: str, info: List[Dict], docs: List[Document]) -> str:
        """Answer 'what is' or 'what are' questions"""
        definitions = []
        
        for item in info:
            content = item['content']
            # Look for definition patterns
            if ':' in content:
                lines = content.split('\n')
                for line in lines:
                    if ':' in line and any(word in question for word in line.lower().split() if len(word) > 3):
                        definitions.append(f"From {item['doc_name']}: {line.strip()}")
        
        if definitions:
            intro = "Based on your company's HR policies:\n\n"
            return intro + "\n".join(definitions[:3])
        else:
            # Provide a contextual answer based on content
            if info:
                return f"According to {info[0]['doc_name']}, here's what I found:\n\n{info[0]['content'][:300]}...\n\nFor complete details, please refer to the full document."
            return "I couldn't find a specific definition in the available documents. Please check with HR for clarification."
    
    def _answer_procedural_question(self, question: str, info: List[Dict], docs: List[Document]) -> str:
        """Answer 'how to' or procedural questions"""
        procedures = []
        
        for doc in docs:
            content = doc.content
            # Look for step-by-step procedures
            if "step" in content.lower() or "procedure" in content.lower():
                lines = content.split('\n')
                capturing = False
                current_procedure = []
                
                for line in lines:
                    line_lower = line.lower()
                    # Start capturing if we find relevant procedure
                    if any(keyword in question for keyword in line_lower.split() if len(keyword) > 3):
                        capturing = True
                    
                    if capturing:
                        if "step" in line_lower or re.match(r'^\d+\.', line.strip()):
                            current_procedure.append(line.strip())
                        elif len(current_procedure) > 0 and line.strip() == "":
                            # End of procedure
                            procedures.append({
                                'doc': doc.name,
                                'steps': current_procedure[:5]  # Max 5 steps
                            })
                            capturing = False
                            current_procedure = []
        
        if procedures:
            answer = "Here's the process based on your company policies:\n\n"
            for proc in procedures[:1]:  # Take first procedure found
                answer += f"According to {proc['doc']}:\n"
                for step in proc['steps']:
                    answer += f"• {step}\n"
            return answer
        else:
            # Provide general guidance
            return f"For detailed procedures, please refer to {docs[0].name if docs else 'the HR handbook'}. You can also contact your HR representative for step-by-step guidance on this process."
    
    def _answer_temporal_question(self, question: str, info: List[Dict], docs: List[Document]) -> str:
        """Answer 'when' questions about timing, dates, deadlines"""
        timings = []
        
        for doc in docs:
            content = doc.content.lower()
            # Look for time-related information
            patterns = [
                r"(\d+)\s*(?:days?|weeks?|months?|hours?)",
                r"(?:by|before|within|after)\s*(\w+\s*\d+|\d+\s*\w+)",
                r"(?:january|february|march|april|may|june|july|august|september|october|november|december)",
                r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                r"(?:annual|quarterly|monthly|weekly|daily)"
            ]
            
            for pattern in patterns:
                if re.search(pattern, content):
                    # Extract surrounding context
                    sentences = content.split('.')
                    for sent in sentences:
                        if re.search(pattern, sent) and any(word in question for word in sent.split() if len(word) > 3):
                            timings.append(f"From {doc.name}: {sent.strip()}")
        
        if timings:
            return "Regarding timing:\n\n" + "\n".join(f"• {timing}" for timing in timings[:3])
        else:
            return "I couldn't find specific timing information for your query. Please check the relevant policy documents or contact HR for exact dates and deadlines."
    
    def _answer_person_question(self, question: str, info: List[Dict], docs: List[Document]) -> str:
        """Answer 'who' questions about responsibilities, contacts"""
        responsibilities = []
        
        for doc in docs:
            content = doc.content
            # Look for role/responsibility patterns
            patterns = [
                r"(?:manager|supervisor|hr|employee|department|team lead)",
                r"(?:responsible for|contact|report to|approved by)",
                r"(?:authority|accountability|ownership)"
            ]
            
            lines = content.split('\n')
            for line in lines:
                if any(re.search(pattern, line.lower()) for pattern in patterns):
                    if any(word in question for word in line.lower().split() if len(word) > 3):
                        responsibilities.append(f"From {doc.name}: {line.strip()}")
        
        if responsibilities:
            return "Based on company policies:\n\n" + "\n".join(f"• {resp}" for resp in responsibilities[:3])
        else:
            return "For specific contact information and responsibilities, please refer to your organizational chart or contact HR directly."
    
    def _answer_policy_question(self, question: str, info: List[Dict], docs: List[Document]) -> str:
        """Answer questions about policies, rules, guidelines"""
        policies = []
        
        for item in info:
            # Extract policy statements
            content = item['content']
            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(word in line_lower for word in ['must', 'shall', 'required', 'mandatory', 'policy', 'rule']):
                    policies.append(f"From {item['doc_name']}: {line.strip()}")
        
        if policies:
            return "According to company policy:\n\n" + "\n".join(f"• {policy}" for policy in policies[:4])
        else:
            if info:
                return f"Based on {info[0]['doc_name']}:\n\n{info[0]['content'][:400]}...\n\nPlease review the complete policy document for full details."
            return "Please refer to the relevant policy documents or contact HR for specific policy information."
    
    def _synthesize_general_answer(self, question: str, info: List[Dict], docs: List[Document]) -> str:
        """Synthesize a general answer from available information"""
        if not info:
            return "I couldn't find specific information about your query. Please contact HR for assistance."
        
        # Create a comprehensive answer
        answer_parts = []
        
        # Start with the most relevant information
        primary_info = info[0]
        answer_parts.append(f"Based on {primary_info['doc_name']}:")
        answer_parts.append(f"\n{primary_info['content'][:400]}")
        
        # Add supporting information from other documents
        if len(info) > 1:
            answer_parts.append("\n\nAdditional relevant information:")
            for item in info[1:3]:  # Max 2 additional sources
                snippet = item['content'][:200]
                answer_parts.append(f"\n• From {item['doc_name']}: {snippet}...")
        
        # Add actionable conclusion
        answer_parts.append("\n\nFor complete details and specific requirements, please refer to the full policy documents or contact your HR representative.")
        
        return "\n".join(answer_parts)

def load_sample_documents():
    """Load sample HR documents for demonstration"""
    if st.session_state.sample_docs_loaded:
        return
    
    sample_docs = [
        Document(
            id="sample1",
            name="Employee_Handbook_2024.pdf",
            content="""
            EMPLOYEE HANDBOOK 2024
            
            1. INTRODUCTION
            Welcome to our company. This handbook outlines key policies and procedures for all employees.
            
            2. WORKING HOURS
            Standard working hours: 40 hours per week, Monday to Friday, 9 AM to 6 PM.
            Flexible working arrangements may be available upon manager approval.
            Core hours are from 10 AM to 4 PM when all employees must be available.
            
            3. LEAVE POLICY
            Annual Leave: 21 days per year for full-time employees.
            Sick Leave: 10 days per year with medical certificate required after 3 consecutive days.
            Personal Leave: 3 days per year for personal emergencies.
            Maternity Leave: 12 weeks paid leave for eligible employees.
            Paternity Leave: 2 weeks paid leave for new fathers.
            Leave requests must be submitted at least 2 weeks in advance for planned leave.
            
            4. CODE OF CONDUCT
            All employees must maintain professional behavior and adhere to company ethics.
            Discrimination, harassment, and unprofessional conduct will not be tolerated.
            Violations will result in disciplinary action up to and including termination.
            
            5. REMOTE WORK POLICY
            Employees may request remote work arrangements with manager approval.
            Remote work requires adherence to communication protocols and availability during core hours.
            Equipment will be provided for approved remote workers.
            Maximum of 2 days per week remote work for most positions.
            
            6. BENEFITS
            - Health Insurance: Comprehensive medical, dental, and vision coverage
            - Retirement: 401(k) with company matching up to 5% of base salary
            - Life Insurance: Basic coverage of 2x annual salary provided
            - Professional Development: Annual training budget of $2,000 per employee
            - Gym Membership: $50 monthly reimbursement
            
            7. PERFORMANCE REVIEWS
            Annual performance reviews conducted in December.
            Mid-year check-ins conducted in June.
            Performance ratings directly impact bonus and salary increases.
            
            8. GRIEVANCE PROCEDURE
            Step 1: Discuss concern with immediate supervisor within 5 days
            Step 2: File formal complaint with HR if unresolved within 10 days
            Step 3: Mediation session arranged if necessary
            Step 4: Final review by senior management
            
            9. TERMINATION POLICY
            Notice Period: 2 weeks for employees, 4 weeks for management positions
            Exit interviews required for all departing employees
            Final paycheck includes unused vacation days
            """,
            type="pdf",
            upload_date=datetime.now() - timedelta(days=30),
            metadata={"pages": 15, "department": "HR", "version": "2024.1"}
        ),
        Document(
            id="sample2",
            name="IT_Security_Policy.pdf",
            content="""
            IT SECURITY POLICY
            
            1. DATA PROTECTION
            All company data must be handled according to security protocols.
            Confidential information must not be shared externally without authorization.
            Data classification: Public, Internal, Confidential, Restricted.
            
            2. PASSWORD POLICY
            Passwords must be minimum 12 characters with complexity requirements.
            Must include uppercase, lowercase, numbers, and special characters.
            Password changes required every 90 days.
            No password reuse for last 12 passwords.
            
            3. REMOTE ACCESS
            VPN required for all remote connections to company network.
            Two-factor authentication mandatory for all remote access.
            Personal devices must be approved and enrolled in MDM system.
            
            4. EMAIL USAGE
            Company email for business purposes only.
            No personal use of company email systems.
            All emails are subject to monitoring and retention policies.
            Suspicious emails must be reported to IT immediately.
            
            5. INCIDENT REPORTING
            Security incidents must be reported immediately to IT department.
            Data breaches must be reported within 24 hours.
            Do not attempt to resolve security issues independently.
            
            6. DEVICE MANAGEMENT
            Company devices must be encrypted with BitLocker or FileVault.
            Personal devices require approval for business use.
            Lost or stolen devices must be reported within 4 hours.
            Regular security updates must be installed within 48 hours of release.
            """,
            type="pdf",
            upload_date=datetime.now() - timedelta(days=15),
            metadata={"pages": 8, "department": "IT", "version": "2024.2"}
        ),
        Document(
            id="sample3",
            name="Compensation_Benefits_Guide.pdf",
            content="""
            COMPENSATION AND BENEFITS GUIDE
            
            1. SALARY STRUCTURE
            Base salary determined by role, experience, and market rates.
            Annual salary reviews conducted in January with changes effective February 1st.
            Salary bands reviewed quarterly against market data.
            
            2. BONUSES
            Performance bonuses: Up to 20% of base salary based on individual performance.
            Company bonus: Additional 5-10% based on company performance.
            Referral bonuses: $2,000 for successful hires who complete probation.
            Spot bonuses: Up to $500 for exceptional contributions.
            
            3. HEALTH BENEFITS
            Medical: 100% premium coverage for employees, 80% for dependents
            Dental: 100% preventive care, 80% major procedures, $2000 annual maximum
            Vision: Annual eye exam and $200 allowance for glasses/contacts
            FSA: Up to $2,850 pre-tax contribution allowed
            
            4. RETIREMENT BENEFITS
            401(k) immediate eligibility upon hire
            Company matches 100% of first 3%, 50% of next 2%
            Vesting schedule: 25% per year, fully vested after 4 years
            Annual retirement planning consultations available
            
            5. TIME OFF BENEFITS
            Annual Leave: 20 days per year (note: may differ from handbook)
            Sick Leave: 10 days per year
            Holidays: 10 federal holidays plus 2 floating holidays
            Sabbatical: 4 weeks after 5 years of service
            
            6. ADDITIONAL PERKS
            Gym membership reimbursement: $50/month
            Commuter benefits: $100/month pre-tax
            Employee assistance program: 24/7 support
            Tuition reimbursement: Up to $5,000/year for approved programs
            """,
            type="pdf",
            upload_date=datetime.now() - timedelta(days=7),
            metadata={"pages": 12, "department": "HR", "version": "2024.1"}
        )
    ]
    
    for doc in sample_docs:
        doc.embeddings = DocumentProcessor.create_embeddings(doc.content)
        st.session_state.documents[doc.id] = doc
    
    st.session_state.sample_docs_loaded = True
    save_documents()

def render_employee_portal():
    """Render the employee Q&A interface"""
    st.markdown("<h2 style='color: #1e1e1e;'>👤 Employee Portal - Ask HR Questions</h2>", unsafe_allow_html=True)
    
    # Load sample documents if not already loaded
    if not st.session_state.documents and not st.session_state.sample_docs_loaded:
        load_sample_documents()
    
    # Display available documents info
    if st.session_state.documents:
        with st.expander("📚 Available HR Documents", expanded=False):
            for doc in st.session_state.documents.values():
                st.markdown(f"• **{doc.name}** - Uploaded: {doc.upload_date.strftime('%Y-%m-%d')}")
    
    # Chat interface
    st.markdown("### 💬 Chat with HR Assistant")
    st.markdown("Ask any questions about HR policies, benefits, leave, or other workplace matters.")
    
    # Sample questions for guidance
    with st.expander("💡 Sample Questions", expanded=False):
        st.markdown("""
        • How many days of annual leave do I have?
        • What is the remote work policy?
        • How do I submit a leave request?
        • What health benefits are available?
        • When are performance reviews conducted?
        • What is the notice period for resignation?
        • How much is the referral bonus?
        """)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>You:</b> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                citations = ""
                if message.get("citations"):
                    citations = f"<br><small style='color: #6c757d;'>📎 Sources: {', '.join(message['citations'])}</small>"
                
                # Format the answer with better styling
                formatted_content = message["content"].replace('\n', '<br>')
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>HR Assistant:</b><br>
                    {formatted_content}{citations}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input("Type your question here...", 
                                      placeholder="e.g., How many days of annual leave do I have?",
                                      label_visibility="collapsed")
        with col2:
            submit = st.form_submit_button("Send 📤", use_container_width=True)
        
        if submit and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            assistant = HRAssistant()
            answer, citations = assistant.answer_question(user_input, st.session_state.documents)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
            
            save_chat_history()
            st.rerun()
    
    # Quick action buttons
    st.markdown("### 🎯 Quick Questions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📅 Leave Policy", use_container_width=True):
            question = "What is the company's leave policy? How many days of leave do I get?"
            st.session_state.chat_history.append({"role": "user", "content": question})
            assistant = HRAssistant()
            answer, citations = assistant.answer_question(question, st.session_state.documents)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
            save_chat_history()
            st.rerun()
    
    with col2:
        if st.button("🏥 Health Benefits", use_container_width=True):
            question = "What health benefits are available? What is covered?"
            st.session_state.chat_history.append({"role": "user", "content": question})
            assistant = HRAssistant()
            answer, citations = assistant.answer_question(question, st.session_state.documents)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
            save_chat_history()
            st.rerun()
    
    with col3:
        if st.button("🏠 Remote Work", use_container_width=True):
            question = "What is the remote work policy? How many days can I work from home?"
            st.session_state.chat_history.append({"role": "user", "content": question})
            assistant = HRAssistant()
            answer, citations = assistant.answer_question(question, st.session_state.documents)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
            save_chat_history()
            st.rerun()
    
    with col4:
        if st.button("💰 Compensation", use_container_width=True):
            question = "What are the bonus policies and salary review process?"
            st.session_state.chat_history.append({"role": "user", "content": question})
            assistant = HRAssistant()
            answer, citations = assistant.answer_question(question, st.session_state.documents)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
            save_chat_history()
            st.rerun()
    
    # Clear chat history button
    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = []
        save_chat_history()
        st.success("Chat history cleared!")
        st.rerun()

def render_admin_dashboard():
    """Render the HR admin dashboard"""
    st.markdown("<h2 style='color: #1e1e1e;'>🔐 HR Admin Dashboard</h2>", unsafe_allow_html=True)
    
    # Authentication
    if not st.session_state.authenticated:
        st.warning("Please login to access the admin dashboard")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
            
            if login_btn:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        st.info(f"Demo credentials - Username: {ADMIN_USERNAME}, Password: {ADMIN_PASSWORD}")
        return
    
    # Admin interface
    st.success("Logged in as HR Administrator")
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Statistics
    st.markdown("### 📊 Dashboard Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>📄</h3>
            <h2>{len(st.session_state.documents)}</h2>
            <p>Total Documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_queries = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.markdown(f"""
        <div class="stat-card">
            <h3>💬</h3>
            <h2>{total_queries}</h2>
            <p>Total Queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Count actual policy issues
        total_issues = 0
        if st.session_state.documents:
            analyzer = PolicyAnalyzer()
            for doc in st.session_state.documents.values():
                issues = analyzer.analyze_document(doc, st.session_state.documents)
                total_issues += len(issues)
        
        st.markdown(f"""
        <div class="stat-card">
            <h3>⚠️</h3>
            <h2>{total_issues}</h2>
            <p>Policy Issues</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        health_score = max(0, 100 - (total_issues * 5))  # Deduct 5% per issue
        st.markdown(f"""
        <div class="stat-card">
            <h3>✅</h3>
            <h2>{health_score}%</h2>
            <p>Policy Health</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Document Management")
        
        # Load sample docs button
        if st.button("📚 Load Sample HR Documents", use_container_width=True):
            load_sample_documents()
            st.success("Sample documents loaded successfully!")
            st.rerun()
        
        # File upload
        uploaded_file = st.file_uploader("Upload HR Document", 
                                        type=['pdf', 'txt'],
                                        help="Upload PDF or TXT files")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    if uploaded_file.type == "application/pdf":
                        doc = DocumentProcessor.process_pdf(
                            uploaded_file.read(),
                            uploaded_file.name
                        )
                    elif uploaded_file.type == "text/plain":
                        doc = DocumentProcessor.process_text(
                            uploaded_file.read(),
                            uploaded_file.name
                        )
                    else:
                        doc = None
                        st.error("Unsupported file type")
                    
                    if doc:
                        doc.embeddings = DocumentProcessor.create_embeddings(doc.content)
                        st.session_state.documents[doc.id] = doc
                        save_documents()
                        st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                        st.rerun()
        
        # Document list
        st.markdown("#### Current Documents")
        if st.session_state.documents:
            for doc_id, doc in st.session_state.documents.items():
                with st.container():
                    st.markdown(f"""
                    <div class="document-card">
                        <b>{doc.name}</b><br>
                        <small>Type: {doc.type.upper()} | 
                        Uploaded: {doc.upload_date.strftime('%Y-%m-%d %H:%M')}</small><br>
                        <small>ID: {doc.id}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button(f"📊 Analyze", key=f"analyze_{doc_id}"):
                            st.session_state.selected_doc_for_analysis = doc_id
                    with col_b:
                        if st.button(f"👁️ View", key=f"view_{doc_id}"):
                            with st.expander(f"Document Content: {doc.name}"):
                                st.text(doc.content[:2000] + "..." if len(doc.content) > 2000 else doc.content)
                    with col_c:
                        if st.button(f"🗑️ Delete", key=f"delete_{doc_id}"):
                            del st.session_state.documents[doc_id]
                            save_documents()
                            st.rerun()
        else:
            st.info("No documents uploaded yet. Load sample documents or upload your own.")
        
        # Clear all documents button
        if st.session_state.documents:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.documents = {}
                st.session_state.sample_docs_loaded = False
                save_documents()
                st.success("All documents cleared!")
                st.rerun()
    
    with col2:
        st.markdown("### 🔍 Policy Health Analysis")
        
        if hasattr(st.session_state, 'selected_doc_for_analysis') and st.session_state.selected_doc_for_analysis:
            selected_doc = st.session_state.documents.get(st.session_state.selected_doc_for_analysis)
            if selected_doc:
                st.markdown(f"**Analyzing:** {selected_doc.name}")
                
                with st.spinner("Analyzing document for policy issues..."):
                    analyzer = PolicyAnalyzer()
                    issues = analyzer.analyze_document(selected_doc, st.session_state.documents)
                
                if issues:
                    st.markdown(f"#### Found {len(issues)} Issues")
                    
                    high_issues = [i for i in issues if i.severity == "high"]
                    medium_issues = [i for i in issues if i.severity == "medium"]
                    low_issues = [i for i in issues if i.severity == "low"]
                    
                    for issue in high_issues:
                        st.markdown(f"""
                        <div class="policy-health-card severity-high">
                            <b>🔴 HIGH: {issue.type.replace('_', ' ').title()}</b><br>
                            <span style='color: #1e1e1e;'>{issue.description}</span><br>
                            <small style='color: #6c757d;'><b>Fix:</b> {issue.suggested_fix}</small><br>
                            <small style='color: #6c757d;'><b>Sources:</b> {', '.join(issue.source_docs)}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    for issue in medium_issues:
                        st.markdown(f"""
                        <div class="policy-health-card severity-medium">
                            <b>🟡 MEDIUM: {issue.type.replace('_', ' ').title()}</b><br>
                            <span style='color: #1e1e1e;'>{issue.description}</span><br>
                            <small style='color: #6c757d;'><b>Fix:</b> {issue.suggested_fix}</small><br>
                            <small style='color: #6c757d;'><b>Sources:</b> {', '.join(issue.source_docs)}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    for issue in low_issues[:3]:  # Limit to 3 low issues
                        st.markdown(f"""
                        <div class="policy-health-card severity-low">
                            <b>🟢 LOW: {issue.type.replace('_', ' ').title()}</b><br>
                            <span style='color: #1e1e1e;'>{issue.description}</span><br>
                            <small style='color: #6c757d;'><b>Fix:</b> {issue.suggested_fix}</small><br>
                            <small style='color: #6c757d;'><b>Sources:</b> {', '.join(issue.source_docs)}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Export report button
                    if st.button("📥 Export Health Report"):
                        report = {
                            "document": selected_doc.name,
                            "analysis_date": datetime.now().isoformat(),
                            "total_issues": len(issues),
                            "issues": [
                                {
                                    "severity": issue.severity,
                                    "type": issue.type,
                                    "description": issue.description,
                                    "suggested_fix": issue.suggested_fix,
                                    "sources": issue.source_docs
                                }
                                for issue in issues
                            ]
                        }
                        st.download_button(
                            label="Download JSON Report",
                            data=json.dumps(report, indent=2),
                            file_name=f"policy_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                else:
                    st.success("✅ No policy issues detected!")
        else:
            st.info("Select a document from the left panel to analyze its policy health.")
        
        # Contradiction Matrix
        if len(st.session_state.documents) > 1:
            st.markdown("#### 🔄 Cross-Document Contradictions")
            
            all_contradictions = []
            for doc1_id, doc1 in st.session_state.documents.items():
                for doc2_id, doc2 in st.session_state.documents.items():
                    if doc1_id < doc2_id:
                        analyzer = PolicyAnalyzer()
                        contradictions = analyzer._find_contradictions(doc1, doc2)
                        all_contradictions.extend(contradictions)
            
            if all_contradictions:
                st.warning(f"⚠️ Found {len(all_contradictions)} contradictions between documents")
                for contradiction in all_contradictions[:3]:
                    st.markdown(f"""
                    <div class="policy-health-card severity-high">
                        <b style='color: #1e1e1e;'>Contradiction Detected</b><br>
                        <span style='color: #1e1e1e;'>{contradiction.description}</span><br>
                        <small style='color: #6c757d;'><b>Documents:</b> {', '.join(contradiction.source_docs)}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No contradictions found between documents")

def main():
    """Main application"""
    # Header with light theme
    # Render header using `st.image()` for the logo and columns for layout
    logo_path = "logo.png"
    col1, col2 = st.columns([1, 8])
    with col1:
        if os.path.exists(logo_path):
            try:
                st.image(logo_path, width=64)
            except Exception:
                st.markdown("<h1>👥</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1>👥</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='padding-left:8px'>
            <h1 style='margin:0; padding-top:6px;'>THE HR - HR Intelligent Assistant</h1>
            <p style='margin:0; color:#e9f3ea;'>AI-Powered HR Policy Management & Employee Support System</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check for missing dependencies
    if not PDF_SUPPORT:
        st.warning("⚠️ PDF support is limited. To enable full PDF processing, install PyPDF2: `pip install PyPDF2`")
    
    # Create tabs
    tab1, tab2 = st.tabs(["👤 Employee Portal", "🔐 Admin Dashboard"])
    
    with tab1:
        render_employee_portal()
    
    with tab2:
        render_admin_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; color: #6c757d;'>
        <small>THE HR Assistant v3.0 - Enhanced AI Edition</small><br>
        <small>✨ Features: Smart document analysis, contextual answers, and persistent storage</small><br>
        <small>Built with Streamlit, LangChain, and Groq API</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()