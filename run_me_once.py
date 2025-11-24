"""
Vector Store Builder
Run this script once to build the FAISS vector store from PDF documents.

Usage:
    python run_me_once.py --pdf data/Attention.pdf
    python run_me_once.py --pdf data/document.pdf --output artifacts/custom_store
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from app.rag_pipeline import build_vector_store, get_vector_store_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FAISS vector store from PDF documents"
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/Attention.pdf",
        help="Path to PDF file (default: data/Attention.pdf)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/vector_store",
        help="Output path for vector store (default: artifacts/vector_store)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if vector store exists"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about existing vector store and exit"
    )
    
    return parser.parse_args()


def validate_pdf_path(pdf_path: str) -> None:
    """
    Validate that PDF file exists.
    
    Args:
        pdf_path: Path to PDF file
    
    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If file is not a PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    # Check if file is readable
    if not os.access(pdf_path, os.R_OK):
        raise PermissionError(f"Cannot read PDF file: {pdf_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Show vector store info if requested
    if args.info:
        logger.info(f"Checking vector store at: {args.output}")
        info = get_vector_store_info(args.output)
        
        if info.get("exists"):
            if "error" in info:
                logger.error(f"Error reading vector store: {info['error']}")
                sys.exit(1)
            else:
                logger.info("📊 Vector Store Information:")
                logger.info(f"  - Location: {info['path']}")
                logger.info(f"  - Documents: {info['num_documents']}")
        else:
            logger.info("❌ Vector store does not exist")
            logger.info(f"   Run: python run_me_once.py --pdf {args.pdf}")
        
        sys.exit(0)
    
    # Validate PDF path
    try:
        logger.info(f"📄 PDF Path: {args.pdf}")
        validate_pdf_path(args.pdf)
        logger.info("✅ PDF file validated")
    except (FileNotFoundError, ValueError, PermissionError) as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    
    # Check if vector store already exists
    if os.path.exists(args.output) and not args.force:
        logger.warning(f"⚠️  Vector store already exists at: {args.output}")
        logger.info("   Use --force to rebuild")
        logger.info("   Use --info to see current store information")
        
        response = input("Rebuild anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            sys.exit(0)
    
    # Create output directory
    output_dir = Path(args.output).parent
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📁 Output Directory: {output_dir}")
    
    # Build vector store
    try:
        logger.info("=" * 60)
        logger.info("🚀 Starting vector store creation...")
        logger.info("=" * 60)
        
        build_vector_store(args.pdf, args.output)
        
        logger.info("=" * 60)
        logger.info("✅ Vector store created successfully!")
        logger.info("=" * 60)
        
        # Show info about created store
        info = get_vector_store_info(args.output)
        if info.get("exists") and "num_documents" in info:
            logger.info(f"📊 Documents indexed: {info['num_documents']}")
        
        logger.info(f"📍 Location: {args.output}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Start the API: uvicorn app.main:app --reload")
        logger.info("  2. Visit: http://localhost:8000/docs")
        logger.info("  3. Try the /ask endpoint")
        
    except Exception as e:
        logger.error(f"❌ Failed to build vector store: {e}")
        logger.error("   Check the error above and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
