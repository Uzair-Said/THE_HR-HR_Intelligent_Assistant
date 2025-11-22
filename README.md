# THE HR - HR Intelligent Assistant 👥

An AI-powered HR policy management system that helps employees quickly understand HR policies and enables HR teams to maintain, check, and improve policy documents through an intuitive interface.

## 🎯 Features

### Employee Portal
- **Natural Language Q&A**: Ask questions about HR policies in plain English
- **Smart Document Retrieval**: AI finds and cites relevant policy documents
- **Quick Action Buttons**: Common queries like leave policy, benefits, and remote work
- **Chat History**: Track all conversations with the HR assistant
- **Multi-Document Reasoning**: Combines information from multiple sources

### Admin Dashboard (HR Team)
- **Secure Access**: Password-protected admin interface
- **Document Management**: Upload, manage, and delete HR documents
- **Policy Health Analysis**: Automated detection of:
  - Policy contradictions between documents
  - Missing essential clauses
  - Ambiguous language
  - Compliance gaps
- **Cross-Document Validation**: Identifies conflicts across multiple policies
- **Health Reports**: Export detailed analysis reports in JSON format
- **Dashboard Statistics**: Track documents, queries, and policy health metrics

## 🚀 Quick Start

### Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run hr_assistant.py
```

4. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

## 📱 How to Use

### For Employees
1. Click on the **Employee Portal** tab
2. View available HR documents in the expandable section
3. Type your question in the chat interface or use quick action buttons
4. Receive AI-powered answers with document citations

### For HR Admins
1. Click on the **Admin Dashboard** tab
2. Login with credentials:
   - Username: `hradmin`
   - Password: `hrpass123`
3. **Load Sample Documents**: Click "Load Sample HR Documents" for demo data
4. **Upload Documents**: Add your own HR PDFs
5. **Analyze Policies**: Select any document and click "Analyze" to check for issues
6. **Export Reports**: Download policy health analysis as JSON

## 🎮 Demo Features

### Sample Documents Included
- **Employee Handbook 2024**: Comprehensive policies including leave, benefits, code of conduct
- **IT Security Policy**: Data protection and security guidelines
- **Compensation & Benefits Guide**: Detailed salary and benefits information

### Pre-configured Scenarios
The system includes intentional policy conflicts for demonstration:
- Conflicting annual leave days between documents
- Missing clauses detection
- Ambiguous language identification

## 🔧 Technical Architecture

### Core Components

1. **Document Processing**
   - PDF text extraction using PyPDF2
   - Simulated embeddings for document indexing
   - In-memory vector storage for quick retrieval

2. **AI Question Answering**
   - Document retrieval based on query relevance
   - Context-aware answer generation
   - Multi-document information synthesis
   - Citation tracking for transparency

3. **Policy Analysis Engine**
   - Pattern-based contradiction detection
   - Required clause verification
   - Ambiguity scoring
   - Severity classification (High/Medium/Low)

4. **User Interface**
   - Streamlit-based responsive design
   - Real-time chat interface
   - Tabbed navigation for role separation
   - Custom CSS for professional appearance

## 📊 Policy Health Metrics

The system analyzes documents for:

### High Severity Issues
- Direct contradictions between documents
- Critical missing policies
- Compliance violations

### Medium Severity Issues
- Missing standard sections
- Incomplete procedures
- Unclear responsibilities

### Low Severity Issues
- Ambiguous language
- Vague timeframes
- Undefined terms

## 🛡️ Security Features

- Password-protected admin access
- Role-based interface separation
- Secure document handling
- Session-based authentication

## 🎯 Use Cases

1. **Employee Self-Service**
   - Quick policy lookups
   - Benefits information
   - Leave balance queries
   - Procedure clarification

2. **HR Document Audit**
   - Policy consistency checks
   - Compliance verification
   - Gap analysis
   - Version control

3. **Policy Improvement**
   - Identify ambiguous language
   - Find missing clauses
   - Detect contradictions
   - Generate improvement reports

## 💡 Why It's Hackathon-Ready

- **No deployment needed**: Runs locally with Streamlit
- **Pre-loaded demo data**: Instant demonstration capability
- **Visual impact**: Professional UI with clear value proposition
- **Real-world application**: Solves actual HR challenges
- **Scalable architecture**: Ready for production enhancement

## 🔄 Future Enhancements

- Integration with actual LLMs (Groq API ready)
- OCR support for scanned documents
- Advanced embedding models
- Department-specific policy filtering
- Multi-language support
- Audit trail and compliance reporting
- Integration with HRIS systems

## 📝 Notes

- The current implementation uses simulated AI responses for demonstration
- Groq API key is included but can be integrated for actual LLM calls
- Document embeddings are simplified for the demo
- Production deployment would require proper database integration

## 🤝 Contributing

This is a hackathon project designed for demonstration. Feel free to extend and improve!

## 📄 License

MIT License - Free to use and modify

---

**Built with ❤️ for making HR processes smarter and more efficient**
