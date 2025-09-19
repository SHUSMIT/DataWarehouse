# GenAI Data Warehouse Platform

**Brain-Inspired Intelligent Data Processing with SpikingBrain Architecture**

A comprehensive full-stack platform that combines SpikingBrain-inspired neural architectures with traditional data warehousing, featuring automated OCR, NLP analysis, and PostgreSQL integration.

## 🧠 Overview

This platform integrates cutting-edge brain-inspired computing principles from the SpikingBrain research with practical data warehouse operations. It automatically processes documents (PDFs, images, structured data) using OCR and advanced NLP models, then stores and analyzes the data in PostgreSQL with intelligent ETL pipelines.

### Key Features

- **SpikingBrain-Inspired Architecture**: Implements adaptive threshold spiking neurons and event-driven computation
- **Automated OCR Processing**: Extracts text from PDFs and images using Tesseract and PDF2Image
- **Advanced NLP Analysis**: Hugging Face transformers for entity extraction, sentiment analysis, and text classification
- **Dynamic PostgreSQL Integration**: Intelligent table creation with automatic type inference
- **Real-time Processing**: Background job processing with WebSocket status updates
- **Interactive Visualization**: AI-powered insights and chart generation

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  PostgreSQL DB  │
│                 │────│                  │────│                 │
│ • File Upload   │    │ • OCR Processing │    │ • Dynamic Tables│
│ • Visualization │    │ • NLP Analysis   │    │ • ETL Pipeline  │
│ • AI Insights   │    │ • SpikingBrain   │    │ • Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌──────────────────┐
                    │ Hugging Face     │
                    │ Models           │
                    │ • BERT/RoBERTa   │
                    │ • GPT-2          │
                    │ • Sentence Trans.│
                    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 16+
- Docker & Docker Compose
- 8GB+ RAM (for AI models)

### 1. Clone Repository
```bash
git clone https://github.com/your-username/genai-datawarehouse.git
cd genai-datawarehouse
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit database credentials
DATABASE_URL=postgresql://genai_user:genai_password@localhost:5432/genai_warehouse
```

### 3. Start Database
```bash
docker-compose up -d postgres redis
```

### 4. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database and download AI models
python startup.py

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm start
```

### 6. Access Platform
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- Database: localhost:5432

## 🤖 SpikingBrain Integration

Based on the SpikingBrain Technical Report, our platform implements:

### Brain-Inspired Mechanisms
- **Adaptive Threshold Spiking**: Dynamic firing thresholds for optimal neuron activity
- **Event-Driven Computation**: Sparse spike trains for energy-efficient processing
- **Linear Attention Variants**: Hybrid attention mechanisms inspired by neural dynamics
- **Multi-Scale Sparsity**: Network-level (MoE) and neuron-level sparsity

### Spike Encoding Schemes
- **Binary Coding**: `{0,1}` for basic event-driven computation
- **Ternary Coding**: `{-1,0,1}` with excitatory/inhibitory spikes
- **Bitwise Coding**: Compressed temporal representation

### Processing Pipeline
```python
# Adaptive threshold spiking
V_th(x) = (1/k) * mean(abs(x))
s_INT = round(x / V_th(x))

# Spike expansion for neuromorphic hardware
s_INT = Σ(s_t) where s_t ∈ {-1,0,1}
```

## 📊 Data Processing Pipeline

### 1. Upload & OCR
- **Supported Formats**: PDF, PNG, JPG, CSV, Excel, JSON
- **OCR Engine**: Tesseract + PDF2Image for document processing
- **Batch Processing**: Multiple file upload with progress tracking

### 2. GenAI Analysis
- **Entity Recognition**: Extract persons, organizations, locations
- **Sentiment Analysis**: Document emotion classification
- **Content Classification**: Automatic document type detection
- **Data Quality Assessment**: Completeness and consistency scoring

### 3. ETL Pipeline
- **Dynamic Schema Detection**: Automatic column type inference
- **Table Creation**: PostgreSQL tables with optimized indexes
- **Data Transformation**: Type conversion and validation
- **Metadata Storage**: AI-generated insights and statistics

### 4. Visualization & Analytics
- **Interactive Charts**: Bar, line, pie, scatter plots
- **AI Insights**: Correlation detection, anomaly identification
- **Export Capabilities**: CSV, JSON data export

## 🗃️ Database Schema

### Core Tables
```sql
-- File tracking
file_uploads (id, filename, file_path, file_size, status, created_at)

-- Job monitoring  
processing_jobs (id, file_id, job_type, status, result, error_message)

-- Dynamic data tables
data_tables (id, table_name, row_count, column_count, schema, ai_metadata)

-- AI insights
ai_insights (id, table_name, insight_type, content, confidence_score)
```

### Dynamic Data Tables
Tables are created automatically based on uploaded data:
- **Intelligent Type Detection**: INTEGER, FLOAT, VARCHAR, JSON, TIMESTAMP
- **Constraint Generation**: NOT NULL, PRIMARY KEY, indexes
- **Metadata Preservation**: Original structure and AI analysis results

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
REDIS_URL=redis://localhost:6379

# AI Models
HUGGINGFACE_API_KEY=your_token_here
OPENAI_API_KEY=your_key_here  # Optional

# Security
SECRET_KEY=your-secret-key
```

### Model Configuration
```python
# Supported Hugging Face models
MODELS = {
    'ner': 'dbmdz/bert-large-cased-finetuned-conll03-english',
    'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'classification': 'facebook/bart-large-mnli',
    'embeddings': 'all-MiniLM-L6-v2'
}
```

## 📈 Performance Optimizations

### SpikingBrain Efficiency Gains
- **69.15% Sparsity**: Sparse spike representation reduces computation
- **Linear Complexity**: O(n) attention mechanisms vs O(n²) traditional
- **Event-Driven**: Only active neurons consume resources
- **Memory Efficiency**: Constant memory usage for long sequences

### Database Optimizations
- **Automatic Indexing**: Performance indexes on frequently queried columns
- **Connection Pooling**: Efficient database connection management
- **Batch Processing**: Bulk insert operations for large datasets
- **Query Optimization**: Intelligent query planning and execution

## 🔬 Research Integration

This platform implements concepts from the **SpikingBrain Technical Report**:

### Architectural Innovations
- **Hybrid Linear Attention**: Combining local and global attention mechanisms
- **MoE Integration**: Mixture-of-Experts for specialized processing
- **Spike-Based Computation**: Integer spike counts with temporal expansion

### Biological Inspiration
- **Adaptive Thresholds**: Dynamic firing based on membrane potential
- **Sparse Activation**: Event-driven computation mimicking neural efficiency
- **Modular Processing**: Specialized expert networks for different data types

## 🚀 Deployment

### Production Deployment
```bash
# Build containers
docker-compose build

# Deploy with production settings
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose up --scale app=3
```

### Cloud Deployment
- **AWS**: ECS/EKS with RDS PostgreSQL
- **Google Cloud**: GKE with Cloud SQL
- **Azure**: AKS with PostgreSQL Flexible Server



## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request


## 🔗 References

- [SpikingBrain Technical Report](https://arxiv.org/abs/2509.05276)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## 🙏 Acknowledgments

- SpikingBrain research team for brain-inspired computing insights
- Hugging Face for transformer model implementations
- FastAPI community for excellent async framework
- PostgreSQL team for robust database engine

---

**Built with brain-inspired intelligence and modern engineering practices** 🧠⚡
