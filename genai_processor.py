# genai_processor.py - GenAI Processing with Hugging Face Models
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    BertTokenizer,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import json
import logging
from datetime import datetime
import asyncio
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

class GenAIProcessor:
    """GenAI processor with Hugging Face models for intelligent data processing"""
    
    def __init__(self):
        self.models_ready = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model containers
        self.ocr_pipeline = None
        self.text_classifier = None
        self.ner_pipeline = None
        self.text_generator = None
        self.embedder = None
        self.data_analyzer = None
        
    async def initialize_models(self):
        """Initialize all AI models"""
        try:
            logger.info("Loading GenAI models...")
            
            # 1. Text Classification for document type detection
            logger.info("Loading text classifier...")
            self.text_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 2. Named Entity Recognition for data extraction
            logger.info("Loading NER model...")
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 3. Text Generation for data insights
            logger.info("Loading text generation model...")
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 4. Sentence Embeddings for similarity and clustering
            logger.info("Loading sentence transformer...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 5. Data Analysis Pipeline
            logger.info("Initializing data analysis pipeline...")
            await self._initialize_data_analyzer()
            
            self.models_ready = True
            logger.info("All GenAI models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    async def _initialize_data_analyzer(self):
        """Initialize specialized data analysis models"""
        self.data_analyzer = {
            'table_classifier': pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            ),
            'data_quality_checker': pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        }
    
    async def analyze_data(self, data: List[Dict]) -> List[Dict]:
        """Main data analysis function with GenAI enhancement"""
        try:
            logger.info(f"Analyzing {len(data)} records with GenAI...")
            
            if not data:
                return data
            
            # 1. Data Quality Analysis
            quality_report = await self._analyze_data_quality(data)
            
            # 2. Content Classification and Enhancement
            enhanced_data = await self._enhance_data_content(data)
            
            # 3. Entity Extraction for text fields
            entity_enhanced_data = await self._extract_entities(enhanced_data)
            
            # 4. Add AI-generated metadata
            final_data = await self._add_ai_metadata(entity_enhanced_data, quality_report)
            
            logger.info(f"GenAI analysis completed. Enhanced {len(final_data)} records")
            return final_data
            
        except Exception as e:
            logger.error(f"Error in GenAI analysis: {str(e)}")
            return data  # Return original data if AI processing fails
    
    async def _analyze_data_quality(self, data: List[Dict]) -> Dict:
        """Analyze data quality using AI"""
        try:
            quality_metrics = {
                'completeness': 0,
                'consistency': 0,
                'anomalies': [],
                'data_types': {},
                'recommendations': []
            }
            
            if not data:
                return quality_metrics
            
            # Calculate completeness
            total_fields = len(data) * len(data[0])
            missing_fields = sum(
                1 for row in data for key, value in row.items() 
                if value is None or value == '' or str(value).lower() == 'nan'
            )
            quality_metrics['completeness'] = (total_fields - missing_fields) / total_fields
            
            # Analyze data types and consistency
            for column in data[0].keys():
                values = [row.get(column) for row in data if row.get(column) is not None]
                if values:
                    quality_metrics['data_types'][column] = await self._infer_data_type(values)
            
            # Generate AI recommendations
            if quality_metrics['completeness'] < 0.8:
                quality_metrics['recommendations'].append(
                    "Consider data cleaning - high percentage of missing values detected"
                )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error in data quality analysis: {str(e)}")
            return {}
    
    async def _infer_data_type(self, values: List) -> str:
        """Infer data type using AI analysis"""
        try:
            # Sample some values for analysis
            sample_values = values[:min(100, len(values))]
            
            # Basic type inference
            numeric_count = sum(1 for v in sample_values if self._is_numeric(v))
            date_count = sum(1 for v in sample_values if self._is_date_like(str(v)))
            
            if numeric_count / len(sample_values) > 0.8:
                return "numeric"
            elif date_count / len(sample_values) > 0.5:
                return "datetime"
            else:
                return "text"
                
        except:
            return "unknown"
    
    def _is_numeric(self, value) -> bool:
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except:
            return False
    
    def _is_date_like(self, value: str) -> bool:
        """Check if value looks like a date"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\w+\s+\d{1,2},\s+\d{4}'
        ]
        return any(re.search(pattern, str(value)) for pattern in date_patterns)
    
    async def _enhance_data_content(self, data: List[Dict]) -> List[Dict]:
        """Enhance data content using GenAI"""
        try:
            enhanced_data = []
            
            for row in data:
                enhanced_row = row.copy()
                
                # Find text columns for AI enhancement
                text_columns = [
                    col for col, val in row.items() 
                    if isinstance(val, str) and len(str(val)) > 10
                ]
                
                # Apply AI analysis to text columns
                for col in text_columns:
                    text_value = str(row[col])
                    
                    # Classify content type
                    content_type = await self._classify_text_content(text_value)
                    enhanced_row[f"{col}_ai_type"] = content_type
                    
                    # Extract sentiment for relevant text
                    sentiment = await self._analyze_sentiment(text_value)
                    enhanced_row[f"{col}_sentiment"] = sentiment
                
                enhanced_data.append(enhanced_row)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error enhancing data content: {str(e)}")
            return data
    
    async def _classify_text_content(self, text: str) -> str:
        """Classify text content type using AI"""
        try:
            if len(text) > 512:
                text = text[:512]  # Truncate for model limits
            
            candidate_labels = [
                "personal_info", "business_data", "description", 
                "address", "communication", "financial", "other"
            ]
            
            result = self.data_analyzer['table_classifier'](text, candidate_labels)
            return result['labels'][0] if result['labels'] else "unknown"
            
        except Exception as e:
            logger.error(f"Error classifying text content: {str(e)}")
            return "unknown"
    
    async def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using AI"""
        try:
            if len(text) > 512:
                text = text[:512]
            
            result = self.data_analyzer['data_quality_checker'](text)
            return {
                "label": result[0]['label'],
                "score": round(result[0]['score'], 3)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "neutral", "score": 0.5}
    
    async def _extract_entities(self, data: List[Dict]) -> List[Dict]:
        """Extract named entities from text data"""
        try:
            enhanced_data = []
            
            for row in data:
                enhanced_row = row.copy()
                
                # Find text columns for entity extraction
                text_columns = [
                    col for col, val in row.items() 
                    if isinstance(val, str) and len(str(val)) > 5
                    and not col.endswith('_ai_type') and not col.endswith('_sentiment')
                ]
                
                extracted_entities = {}
                
                for col in text_columns:
                    text_value = str(row[col])
                    if len(text_value) > 512:
                        text_value = text_value[:512]
                    
                    try:
                        entities = self.ner_pipeline(text_value)
                        if entities:
                            extracted_entities[f"{col}_entities"] = [
                                {
                                    "text": ent["word"],
                                    "label": ent["entity_group"],
                                    "confidence": round(ent["score"], 3)
                                }
                                for ent in entities
                            ]
                    except:
                        continue
                
                enhanced_row.update(extracted_entities)
                enhanced_data.append(enhanced_row)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return data
    
    async def _add_ai_metadata(self, data: List[Dict], quality_report: Dict) -> List[Dict]:
        """Add AI-generated metadata to the dataset"""
        try:
            # Add processing timestamp and AI metadata to each row
            timestamp = datetime.utcnow().isoformat()
            
            enhanced_data = []
            for i, row in enumerate(data):
                enhanced_row = row.copy()
                enhanced_row.update({
                    '_ai_processed_at': timestamp,
                    '_ai_record_id': i + 1,
                    '_ai_quality_score': quality_report.get('completeness', 0),
                    '_ai_anomaly_detected': False  # Placeholder for anomaly detection
                })
                enhanced_data.append(enhanced_row)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error adding AI metadata: {str(e)}")
            return data
    
    async def structure_text(self, text: str, source_name: str) -> List[Dict]:
        """Structure unstructured text using GenAI"""
        try:
            logger.info(f"Structuring text from {source_name} using GenAI...")
            
            # Split text into meaningful chunks
            chunks = self._split_text_intelligently(text)
            
            structured_data = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:
                    continue
                
                # Extract entities and classify content
                entities = self.ner_pipeline(chunk[:512])
                content_type = await self._classify_text_content(chunk)
                
                # Create structured record
                record = {
                    'source': source_name,
                    'chunk_id': i + 1,
                    'text_content': chunk.strip(),
                    'content_type': content_type,
                    'extracted_entities': [
                        {
                            "text": ent["word"],
                            "label": ent["entity_group"],
                            "confidence": round(ent["score"], 3)
                        }
                        for ent in entities
                    ] if entities else [],
                    'processed_at': datetime.utcnow().isoformat()
                }
                
                # Try to extract key-value pairs
                key_value_pairs = self._extract_key_value_pairs(chunk)
                if key_value_pairs:
                    record['extracted_fields'] = key_value_pairs
                
                structured_data.append(record)
            
            logger.info(f"Structured {len(structured_data)} records from text")
            return structured_data
            
        except Exception as e:
            logger.error(f"Error structuring text: {str(e)}")
            # Fallback to basic structure
            return [{
                'source': source_name,
                'text_content': text,
                'processed_at': datetime.utcnow().isoformat()
            }]
    
    def _split_text_intelligently(self, text: str) -> List[str]:
        """Split text into meaningful chunks"""
        # Split by common delimiters and patterns
        patterns = [
            r'\n\s*\n',  # Double newlines
            r'\.\s+[A-Z]',  # Sentence boundaries
            r'(?<=[.!?])\s+(?=[A-Z])',  # After punctuation
        ]
        
        chunks = [text]
        
        for pattern in patterns:
            new_chunks = []
            for chunk in chunks:
                splits = re.split(pattern, chunk)
                new_chunks.extend([s.strip() for s in splits if s.strip()])
            chunks = new_chunks
        
        # Filter out very short chunks and combine if needed
        meaningful_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(chunk) < 50 and current_chunk:
                current_chunk += " " + chunk
            else:
                if current_chunk:
                    meaningful_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            meaningful_chunks.append(current_chunk)
        
        return meaningful_chunks[:50]  # Limit to 50 chunks max
    
    def _extract_key_value_pairs(self, text: str) -> Dict:
        """Extract key-value pairs from text"""
        try:
            patterns = [
                r'([A-Za-z][A-Za-z\s]+?):\s*([^\n:]+)',  # "Key: Value"
                r'([A-Za-z][A-Za-z\s]+?)\s*=\s*([^\n=]+)',  # "Key = Value"
                r'([A-Z][A-Za-z\s]+?)\s+([A-Z0-9][^\n]*)',  # "Key Value"
            ]
            
            extracted = {}
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for key, value in matches:
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    if len(key) > 1 and len(value) > 1:
                        extracted[key] = value
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting key-value pairs: {str(e)}")
            return {}
    
    async def enhance_structured_data(self, data: List[Dict]) -> List[Dict]:
        """Enhance already structured data with AI insights"""
        try:
            logger.info(f"Enhancing {len(data)} structured records with AI...")
            
            # Apply full AI analysis pipeline
            enhanced_data = await self.analyze_data(data)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error enhancing structured data: {str(e)}")
            return data
    
    async def generate_insights(self, data: List[Dict]) -> List[Dict]:
        """Generate AI-powered insights for data"""
        try:
            logger.info("Generating AI insights...")
            
            if not data:
                return []
            
            insights = []
            
            # 1. Data Summary Insights
            summary_insight = await self._generate_summary_insights(data)
            insights.append(summary_insight)
            
            # 2. Pattern Recognition
            pattern_insights = await self._detect_patterns(data)
            insights.extend(pattern_insights)
            
            # 3. Anomaly Detection
            anomaly_insights = await self._detect_anomalies(data)
            insights.extend(anomaly_insights)
            
            # 4. Content Analysis Insights
            content_insights = await self._analyze_content_insights(data)
            insights.extend(content_insights)
            
            # 5. Recommendations
            recommendations = await self._generate_recommendations(data)
            insights.extend(recommendations)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    async def _generate_summary_insights(self, data: List[Dict]) -> Dict:
        """Generate summary insights"""
        try:
            numeric_cols = []
            text_cols = []
            
            if data:
                for col, value in data[0].items():
                    if isinstance(value, (int, float)) or self._is_numeric(value):
                        numeric_cols.append(col)
                    elif isinstance(value, str):
                        text_cols.append(col)
            
            return {
                "type": "summary",
                "title": "Dataset Overview",
                "insights": [
                    f"Dataset contains {len(data)} records",
                    f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:5])}",
                    f"Found {len(text_cols)} text columns: {', '.join(text_cols[:5])}",
                ],
                "confidence": 0.95
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"type": "summary", "title": "Dataset Overview", "insights": [], "confidence": 0}
    
    async def _detect_patterns(self, data: List[Dict]) -> List[Dict]:
        """Detect patterns in data using AI"""
        try:
            patterns = []
            
            if not data:
                return patterns
            
            # Analyze column correlations
            numeric_data = {}
            for col in data[0].keys():
                values = [row.get(col) for row in data]
                numeric_values = [float(v) for v in values if self._is_numeric(v)]
                if len(numeric_values) > len(data) * 0.5:  # At least 50% numeric
                    numeric_data[col] = numeric_values
            
            # Find correlations
            if len(numeric_data) >= 2:
                correlations = []
                cols = list(numeric_data.keys())
                
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        col1, col2 = cols[i], cols[j]
                        if len(numeric_data[col1]) == len(numeric_data[col2]):
                            corr = np.corrcoef(numeric_data[col1], numeric_data[col2])[0, 1]
                            if not np.isnan(corr) and abs(corr) > 0.7:
                                correlations.append({
                                    "columns": [col1, col2],
                                    "correlation": round(corr, 3),
                                    "strength": "strong" if abs(corr) > 0.8 else "moderate"
                                })
                
                if correlations:
                    patterns.append({
                        "type": "correlation",
                        "title": "Strong Correlations Detected",
                        "insights": [f"{c['columns'][0]} and {c['columns'][1]} show {c['strength']} correlation ({c['correlation']})" for c in correlations],
                        "confidence": 0.85
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return []
    
    async def _detect_anomalies(self, data: List[Dict]) -> List[Dict]:
        """Detect anomalies using AI"""
        try:
            anomalies = []
            
            # Simple outlier detection for numeric columns
            for col in data[0].keys():
                values = [row.get(col) for row in data]
                numeric_values = [float(v) for v in values if self._is_numeric(v)]
                
                if len(numeric_values) > 10:  # Need sufficient data
                    mean = np.mean(numeric_values)
                    std = np.std(numeric_values)
                    
                    outliers = [v for v in numeric_values if abs(v - mean) > 2 * std]
                    
                    if outliers:
                        anomalies.append({
                            "type": "anomaly",
                            "title": f"Outliers in {col}",
                            "insights": [
                                f"Found {len(outliers)} outliers in {col}",
                                f"Values significantly different from mean ({round(mean, 2)})"
                            ],
                            "confidence": 0.75
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    async def _analyze_content_insights(self, data: List[Dict]) -> List[Dict]:
        """Analyze content-specific insights"""
        try:
            content_insights = []
            
            # Analyze text content distribution
            text_cols = [col for col in data[0].keys() if isinstance(data[0].get(col), str)]
            
            for col in text_cols[:3]:  # Limit to first 3 text columns
                values = [str(row.get(col, '')) for row in data if row.get(col)]
                
                if values:
                    avg_length = sum(len(v) for v in values) / len(values)
                    unique_ratio = len(set(values)) / len(values)
                    
                    insights_text = [
                        f"Average text length: {round(avg_length, 1)} characters",
                        f"Uniqueness ratio: {round(unique_ratio * 100, 1)}%"
                    ]
                    
                    if unique_ratio < 0.1:
                        insights_text.append("High repetition detected - possible categorical data")
                    
                    content_insights.append({
                        "type": "content",
                        "title": f"Text Analysis for {col}",
                        "insights": insights_text,
                        "confidence": 0.8
                    })
            
            return content_insights
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return []
    
    async def _generate_recommendations(self, data: List[Dict]) -> List[Dict]:
        """Generate AI-powered recommendations"""
        try:
            recommendations = []
            
            # Data quality recommendations
            missing_data = {}
            for col in data[0].keys():
                missing_count = sum(1 for row in data if not row.get(col) or str(row.get(col)).strip() == '')
                if missing_count > 0:
                    missing_data[col] = missing_count
            
            if missing_data:
                high_missing = {k: v for k, v in missing_data.items() if v > len(data) * 0.1}
                if high_missing:
                    recommendations.append({
                        "type": "recommendation",
                        "title": "Data Quality Improvement",
                        "insights": [
                            f"Consider data cleaning for columns with high missing values: {', '.join(high_missing.keys())}",
                            "Missing data may affect analysis accuracy"
                        ],
                        "confidence": 0.9
                    })
            
            # Analysis recommendations
            numeric_cols = [col for col in data[0].keys() if self._is_numeric(data[0].get(col))]
            if len(numeric_cols) >= 2:
                recommendations.append({
                    "type": "recommendation",
                    "title": "Analysis Opportunities",
                    "insights": [
                        f"Multiple numeric columns available for statistical analysis",
                        f"Consider regression analysis between: {', '.join(numeric_cols[:3])}",
                        "Clustering analysis may reveal data patterns"
                    ],
                    "confidence": 0.8
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    async def create_visualization(self, data: List[Dict], chart_type: str, x_axis: str, y_axis: Optional[str]) -> Dict:
        """Create AI-enhanced visualization data"""
        try:
            logger.info(f"Creating {chart_type} visualization for {x_axis} vs {y_axis}")
            
            if not data:
                return {"error": "No data available"}
            
            viz_data = {
                "chart_type": chart_type,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "data": [],
                "metadata": {}
            }
            
            if chart_type == "bar" or chart_type == "pie":
                # Aggregate data by x_axis
                aggregated = {}
                for row in data:
                    x_val = str(row.get(x_axis, 'Unknown'))
                    if y_axis and row.get(y_axis) is not None:
                        y_val = float(row.get(y_axis, 0)) if self._is_numeric(row.get(y_axis)) else 1
                    else:
                        y_val = 1
                    
                    aggregated[x_val] = aggregated.get(x_val, 0) + y_val
                
                viz_data["data"] = [
                    {"label": k, "value": v} for k, v in aggregated.items()
                ]
                
            elif chart_type == "line" or chart_type == "scatter":
                # Use raw data points
                for row in data:
                    x_val = row.get(x_axis)
                    y_val = row.get(y_axis) if y_axis else 1
                    
                    if x_val is not None and y_val is not None:
                        viz_data["data"].append({
                            "x": x_val,
                            "y": float(y_val) if self._is_numeric(y_val) else 0
                        })
            
            # Add AI-generated insights about the visualization
            viz_data["ai_insights"] = await self._generate_viz_insights(viz_data, data)
            
            return viz_data
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_viz_insights(self, viz_data: Dict, original_data: List[Dict]) -> List[str]:
        """Generate AI insights about the visualization"""
        try:
            insights = []
            
            if viz_data["data"]:
                data_points = viz_data["data"]
                
                if viz_data["chart_type"] in ["bar", "pie"]:
                    # Categorical analysis
                    values = [d["value"] for d in data_points]
                    total = sum(values)
                    max_val = max(values)
                    max_item = next(d["label"] for d in data_points if d["value"] == max_val)
                    
                    insights.append(f"'{max_item}' represents {round(max_val/total*100, 1)}% of the total")
                    
                    if len(data_points) > 5:
                        insights.append(f"Dataset shows {len(data_points)} distinct categories")
                
                elif viz_data["chart_type"] in ["line", "scatter"]:
                    # Trend analysis
                    y_values = [d["y"] for d in data_points if isinstance(d["y"], (int, float))]
                    if len(y_values) > 2:
                        trend = "increasing" if y_values[-1] > y_values[0] else "decreasing"
                        insights.append(f"Overall trend appears to be {trend}")
                        
                        volatility = np.std(y_values) if len(y_values) > 1 else 0
                        if volatility > np.mean(y_values) * 0.5:
                            insights.append("High volatility detected in the data")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating viz insights: {str(e)}")
            return []