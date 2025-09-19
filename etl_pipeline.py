# etl_pipeline.py - ETL Pipeline for PostgreSQL
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import re
from database import engine, SessionLocal
from models import DataTable, DataQualityReport

logger = logging.getLogger(__name__)

class ETLPipeline:
    """ETL Pipeline for processing data into PostgreSQL data warehouse"""
    
    def __init__(self):
        self.engine = engine
        self.metadata = MetaData()
        
    async def create_table_from_data(self, data: List[Dict], suggested_name: str) -> str:
        """Create a new table in PostgreSQL from processed data"""
        try:
            logger.info(f"Creating table from {len(data)} records...")
            
            if not data:
                raise ValueError("No data provided")
            
            # Generate table name
            table_name = self._generate_table_name(suggested_name)
            
            # Analyze data structure and create schema
            schema = self._analyze_data_structure(data)
            
            # Create table in PostgreSQL
            await self._create_postgres_table(table_name, schema)
            
            # Insert data
            await self._insert_data(table_name, data, schema)
            
            # Record table metadata
            await self._record_table_metadata(table_name, data, schema)
            
            logger.info(f"Successfully created table: {table_name}")
            return table_name
            
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            raise
    
    def _generate_table_name(self, suggested_name: str) -> str:
        """Generate a valid PostgreSQL table name"""
        # Clean the suggested name
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', suggested_name.lower())
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        
        if not clean_name or clean_name[0].isdigit():
            clean_name = f"table_{clean_name}"
        
        # Check if table exists and create unique name
        base_name = clean_name[:50]  # Limit length
        table_name = base_name
        counter = 1
        
        inspector = inspect(self.engine)
        existing_tables = inspector.get_table_names()
        
        while table_name in existing_tables:
            table_name = f"{base_name}_{counter}"
            counter += 1
        
        return table_name
    
    def _analyze_data_structure(self, data: List[Dict]) -> Dict[str, Dict]:
        """Analyze data structure and determine column types"""
        schema = {}
        
        if not data:
            return schema
        
        # Get all possible columns from all records
        all_columns = set()
        for record in data:
            all_columns.update(record.keys())
        
        for column in all_columns:
            # Collect all values for this column
            values = []
            for record in data:
                value = record.get(column)
                if value is not None and value != '' and str(value).lower() != 'nan':
                    values.append(value)
            
            if not values:
                schema[column] = {'type': 'TEXT', 'nullable': True}
                continue
            
            # Determine the best PostgreSQL type
            pg_type = self._determine_postgres_type(values, column)
            
            schema[column] = {
                'type': pg_type,
                'nullable': len(values) < len(data),
                'sample_values': values[:5],
                'unique_count': len(set(str(v) for v in values))
            }
        
        return schema
    
    def _determine_postgres_type(self, values: List, column_name: str) -> str:
        """Determine the appropriate PostgreSQL data type"""
        try:
            # Check for specific patterns first
            if 'id' in column_name.lower() and all(isinstance(v, (int, str)) for v in values):
                # Try to convert to int
                try:
                    [int(v) for v in values]
                    return 'INTEGER'
                except:
                    pass
            
            # Check for integers
            int_count = 0
            float_count = 0
            
            for value in values:
                try:
                    float_val = float(value)
                    if float_val.is_integer():
                        int_count += 1
                    else:
                        float_count += 1
                except:
                    break
            
            total_numeric = int_count + float_count
            
            # If 80% or more are numeric
            if total_numeric >= len(values) * 0.8:
                if float_count > 0:
                    return 'DOUBLE PRECISION'
                else:
                    # Check range for integer types
                    try:
                        int_values = [int(float(v)) for v in values]
                        max_val = max(int_values)
                        min_val = min(int_values)
                        
                        if -32768 <= min_val and max_val <= 32767:
                            return 'SMALLINT'
                        elif -2147483648 <= min_val and max_val <= 2147483647:
                            return 'INTEGER'
                        else:
                            return 'BIGINT'
                    except:
                        return 'DOUBLE PRECISION'
            
            # Check for dates
            date_count = 0
            for value in values[:min(20, len(values))]:  # Sample first 20
                if self._is_date_like(str(value)):
                    date_count += 1
            
            if date_count >= len(values[:20]) * 0.7:
                return 'TIMESTAMP'
            
            # Check for boolean
            bool_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
            if all(str(v).lower() in bool_values for v in values):
                return 'BOOLEAN'
            
            # Check for JSON
            json_count = 0
            for value in values[:5]:  # Sample first 5
                if isinstance(value, (dict, list)):
                    json_count += 1
                elif isinstance(value, str):
                    try:
                        json.loads(value)
                        json_count += 1
                    except:
                        pass
            
            if json_count > 0:
                return 'JSON'
            
            # Determine text type based on length
            max_length = max(len(str(v)) for v in values)
            
            if max_length <= 255:
                return 'VARCHAR(255)'
            elif max_length <= 1000:
                return 'VARCHAR(1000)'
            else:
                return 'TEXT'
                
        except Exception as e:
            logger.warning(f"Error determining type for column {column_name}: {str(e)}")
            return 'TEXT'
    
    def _is_date_like(self, value: str) -> bool:
        """Check if a string value looks like a date"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\w+\s+\d{1,2},\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
        ]
        
        return any(re.search(pattern, str(value)) for pattern in date_patterns)
    
    async def _create_postgres_table(self, table_name: str, schema: Dict[str, Dict]):
        """Create table in PostgreSQL"""
        try:
            # Build CREATE TABLE statement
            columns_sql = []
            
            for column_name, column_info in schema.items():
                # Escape column name if it's a PostgreSQL reserved word
                safe_column_name = f'"{column_name}"' if column_name.upper() in [
                    'ORDER', 'GROUP', 'SELECT', 'FROM', 'WHERE', 'HAVING', 'LIMIT', 'USER'
                ] else column_name
                
                nullable = "NULL" if column_info['nullable'] else "NOT NULL"
                columns_sql.append(f"{safe_column_name} {column_info['type']} {nullable}")
            
            # Add metadata columns
            columns_sql.extend([
                "_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "_row_id SERIAL PRIMARY KEY"
            ])
            
            create_sql = f"""
            CREATE TABLE "{table_name}" (
                {', '.join(columns_sql)}
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.commit()
            
            logger.info(f"Created PostgreSQL table: {table_name}")
            
        except Exception as e:
            logger.error(f"Error creating PostgreSQL table: {str(e)}")
            raise
    
    async def _insert_data(self, table_name: str, data: List[Dict], schema: Dict[str, Dict]):
        """Insert data into PostgreSQL table"""
        try:
            logger.info(f"Inserting {len(data)} records into {table_name}")
            
            if not data:
                return
            
            # Prepare data for insertion
            processed_data = []
            
            for record in data:
                processed_record = {}
                
                for column_name, column_info in schema.items():
                    value = record.get(column_name)
                    
                    # Convert value based on PostgreSQL type
                    converted_value = self._convert_value_for_postgres(
                        value, column_info['type']
                    )
                    
                    processed_record[column_name] = converted_value
                
                processed_data.append(processed_record)
            
            # Create DataFrame and insert using pandas
            df = pd.DataFrame(processed_data)
            
            # Insert data
            df.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"Successfully inserted {len(processed_data)} records")
            
        except Exception as e:
            logger.error(f"Error inserting data: {str(e)}")
            raise
    
    def _convert_value_for_postgres(self, value: Any, pg_type: str) -> Any:
        """Convert Python value to PostgreSQL-compatible format"""
        try:
            if value is None or value == '' or str(value).lower() in ['nan', 'null']:
                return None
            
            if pg_type in ['INTEGER', 'SMALLINT', 'BIGINT']:
                return int(float(value))
            elif pg_type == 'DOUBLE PRECISION':
                return float(value)
            elif pg_type == 'BOOLEAN':
                if isinstance(value, bool):
                    return value
                str_val = str(value).lower()
                return str_val in ['true', '1', 'yes', 'y', 'on']
            elif pg_type == 'TIMESTAMP':
                if isinstance(value, datetime):
                    return value
                # Try to parse date string
                try:
                    return pd.to_datetime(value)
                except:
                    return str(value)
            elif pg_type == 'JSON':
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                elif isinstance(value, str):
                    try:
                        json.loads(value)  # Validate JSON
                        return value
                    except:
                        return json.dumps(str(value))
                else:
                    return json.dumps(str(value))
            else:
                # TEXT, VARCHAR types
                return str(value)
                
        except Exception as e:
            logger.warning(f"Error converting value {value} to {pg_type}: {str(e)}")
            return str(value) if value is not None else None
    
    async def _record_table_metadata(self, table_name: str, data: List[Dict], schema: Dict[str, Dict]):
        """Record table metadata in the data_tables table"""
        try:
            db = SessionLocal()
            
            # Calculate AI metadata
            ai_metadata = {
                'schema_analysis': schema,
                'data_sample': data[:3] if data else [],
                'creation_timestamp': datetime.utcnow().isoformat(),
                'data_types_detected': {
                    col: info['type'] for col, info in schema.items()
                }
            }
            
            # Create DataTable record
            data_table = DataTable(
                table_name=table_name,
                row_count=len(data),
                column_count=len(schema),
                schema=schema,
                ai_metadata=ai_metadata
            )
            
            db.add(data_table)
            db.commit()
            db.close()
            
            logger.info(f"Recorded metadata for table: {table_name}")
            
        except Exception as e:
            logger.error(f"Error recording table metadata: {str(e)}")
            # Don't raise - table creation should succeed even if metadata recording fails
