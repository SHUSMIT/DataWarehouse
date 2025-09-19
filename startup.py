#!/usr/bin/env python3
import asyncio
import os
import subprocess
import sys
from pathlib import Path

async def setup_database():
    """Setup database with Alembic migrations"""
    print("Setting up database...")
    
    # Run Alembic migrations
    try:
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        print("Database migrations completed successfully")
    except subprocess.CalledProcessError:
        print("Error running database migrations")
        sys.exit(1)

async def download_models():
    """Download required AI models"""
    print("Downloading AI models...")
    
    from genai_processor import GenAIProcessor
    
    try:
        processor = GenAIProcessor()
        await processor.initialize_models()
        print("AI models downloaded and initialized successfully")
    except Exception as e:
        print(f"Error initializing AI models: {e}")
        sys.exit(1)

async def main():
    """Main startup routine"""
    print("Starting GenAI Data Warehouse setup...")
    
    # Create necessary directories
    directories = ["uploads", "logs", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Setup database
    await setup_database()
    
    # Download and initialize AI models
    await download_models()
    
    print("Setup completed successfully!")
    print("You can now run the application with: uvicorn main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    asyncio.run(main())