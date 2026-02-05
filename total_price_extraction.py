"""
Invoice Amount Extraction Module

This module extracts line item amounts from invoice PDF documents
and returns structured JSON output.

Supports both text-based and image-based (scanned) PDFs using
OpenAI's vision capabilities.
"""

import os
import sys
import json
import base64
from pathlib import Path

import fitz  # PyMuPDF
import yaml
from dotenv import load_dotenv
from openai import OpenAI


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_api_key() -> str:
    """Load OpenAI API key from .env file."""
    env_path = Path(__file__).parent / ".env"
    
    # Read the .env file manually since it has non-standard format
    with open(env_path, "r") as f:
        content = f.read()
    
    # Parse the key (format: "OpenAi key : <key>")
    if ":" in content:
        api_key = content.split(":", 1)[1].strip()
        return api_key
    
    raise ValueError("Could not find API key in .env file")


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dicts with page number, extracted text, and whether it has text
    """
    pages_data = []
    
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            has_text = bool(text)
            
            pages_data.append({
                "page_number": page_num,
                "text": text,
                "has_text": has_text
            })
    
    return pages_data


def convert_page_to_image(pdf_path: str, page_num: int, dpi: int = 200) -> str:
    """
    Convert a PDF page to a base64-encoded image.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)
        dpi: Resolution for the image
        
    Returns:
        Base64-encoded PNG image string
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_num - 1]
        
        # Create a matrix for the resolution
        zoom = dpi / 72  # 72 is the default DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render the page as a pixmap
        pixmap = page.get_pixmap(matrix=matrix)
        
        # Get PNG bytes and encode to base64
        png_bytes = pixmap.tobytes("png")
        base64_image = base64.b64encode(png_bytes).decode("utf-8")
        
        return base64_image


def extract_amounts_from_text(
    client: OpenAI,
    model_name: str,
    page_text: str,
    page_number: int,
    pdf_name: str
) -> list[dict]:
    """
    Use OpenAI to extract line item amounts from invoice text.
    
    Args:
        client: OpenAI client instance
        model_name: Name of the model to use
        page_text: Extracted text from the PDF page
        page_number: Page number in the PDF
        pdf_name: Name of the PDF file
        
    Returns:
        List of extracted line items with amounts
    """
    prompt = f"""You are an invoice data extraction assistant. Your task is to extract ALL line items with their amounts from the provided invoice text.

CRITICAL RULES:
1. ONLY extract data that is explicitly present in the document
2. DO NOT generate, infer, or make up any data
3. Extract EVERY line item that has an amount/price associated with it
4. Amounts are always in USD

For each line item, extract:
- line_number: The line/item number as shown on the invoice (use sequential numbering if not explicitly shown)
- description: The item description/name
- amount: The total price/amount for that line item (as a decimal number, e.g., 12.12)
- Quantity: The quantity of items ordered (as a decimal number, e.g., 2 or 1.5)
- Unit_price: The price per unit (as a decimal number, e.g., 6.06)

Return a JSON object with a key "items" containing an array of objects. Each object must have these exact keys:
- "line_number" (integer)
- "description" (string)
- "amount" (number)
- "Quantity" (number)
- "Unit_price" (number)

If Quantity or Unit_price is not visible in the document, use null for that field.
If no line items with amounts are found, return: {{"items": []}}

Invoice text to extract from:
---
{page_text}
---"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction assistant. You only extract data that exists in the provided text. Never generate or infer data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content.strip()
        parsed = json.loads(response_text)
        
        # Extract items from response
        items = parsed.get("items", [])
        
        # Add page and PDF metadata to each item
        for item in items:
            item["_page"] = page_number
            item["_pdf"] = pdf_name
            
            # Ensure amount is a float
            if "amount" in item:
                try:
                    item["amount"] = float(item["amount"])
                except (ValueError, TypeError):
                    item["amount"] = 0.0
            
            # Ensure Quantity is a float or None
            if "Quantity" in item and item["Quantity"] is not None:
                try:
                    item["Quantity"] = float(item["Quantity"])
                except (ValueError, TypeError):
                    item["Quantity"] = None
            
            # Ensure Unit_price is a float or None
            if "Unit_price" in item and item["Unit_price"] is not None:
                try:
                    item["Unit_price"] = float(item["Unit_price"])
                except (ValueError, TypeError):
                    item["Unit_price"] = None
        
        return items
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error during text extraction: {e}", file=sys.stderr)
        return []


def extract_amounts_from_image(
    client: OpenAI,
    base64_image: str,
    page_number: int,
    pdf_name: str
) -> list[dict]:
    """
    Use OpenAI Vision to extract line item amounts from an invoice image.
    
    Args:
        client: OpenAI client instance
        base64_image: Base64-encoded image of the invoice page
        page_number: Page number in the PDF
        pdf_name: Name of the PDF file
        
    Returns:
        List of extracted line items with amounts
    """
    prompt = """You are an invoice data extraction assistant. Your task is to extract ALL line items with their amounts from the provided invoice image.

CRITICAL RULES:
1. ONLY extract data that is explicitly visible in the invoice image
2. DO NOT generate, infer, or make up any data
3. Extract EVERY line item that has an amount/price associated with it
4. Amounts are always in USD

For each line item, extract:
- line_number: The line/item number as shown on the invoice (use sequential numbering if not explicitly shown)
- description: The item description/name exactly as shown
- amount: The total price/amount for that line item (as a decimal number, e.g., 12.12)
- Quantity: The quantity of items ordered (as a decimal number, e.g., 2 or 1.5)
- Unit_price: The price per unit (as a decimal number, e.g., 6.06)

Return a JSON object with a key "items" containing an array of objects. Each object must have these exact keys:
- "line_number" (integer)
- "description" (string)
- "amount" (number)
- "Quantity" (number)
- "Unit_price" (number)

If Quantity or Unit_price is not visible in the invoice, use null for that field.
If no line items with amounts are found, return: {"items": []}

IMPORTANT: Only extract what you can clearly see in the image. Do not guess or approximate."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # Using vision-capable model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content.strip()
        parsed = json.loads(response_text)
        
        # Extract items from response
        items = parsed.get("items", [])
        
        # Add page and PDF metadata to each item
        for item in items:
            item["_page"] = page_number
            item["_pdf"] = pdf_name
            
            # Ensure amount is a float
            if "amount" in item:
                try:
                    item["amount"] = float(item["amount"])
                except (ValueError, TypeError):
                    item["amount"] = 0.0
            
            # Ensure Quantity is a float or None
            if "Quantity" in item and item["Quantity"] is not None:
                try:
                    item["Quantity"] = float(item["Quantity"])
                except (ValueError, TypeError):
                    item["Quantity"] = None
            
            # Ensure Unit_price is a float or None
            if "Unit_price" in item and item["Unit_price"] is not None:
                try:
                    item["Unit_price"] = float(item["Unit_price"])
                except (ValueError, TypeError):
                    item["Unit_price"] = None
        
        return items
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error during image extraction: {e}", file=sys.stderr)
        return []


def extract_invoice_amounts(pdf_path: str) -> dict:
    """
    Main function to extract all amounts from an invoice PDF.
    
    Automatically handles both text-based and image-based (scanned) PDFs.
    
    Args:
        pdf_path: Path to the invoice PDF file
        
    Returns:
        Dictionary with extraction results in the required format
    """
    # Get PDF name without extension
    pdf_name = Path(pdf_path).stem
    
    # Initialize result structure
    result = {
        "status": "success",
        "pdf_name": pdf_name,
        "num_pages": 0,
        "total_items": 0,
        "amounts": [],
        "failed_pages": []
    }
    
    try:
        # Load configuration
        config = load_config()
        model_name = config.get("model", {}).get("name", "gpt-4.1-mini")
        
        # Load API key and initialize client
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)
        
        # Extract text from PDF
        pages_data = extract_text_from_pdf(pdf_path)
        result["num_pages"] = len(pages_data)
        
        # Process each page
        all_items = []
        for page_data in pages_data:
            page_num = page_data["page_number"]
            
            try:
                if page_data["has_text"]:
                    # Use text-based extraction
                    print(f"Page {page_num}: Using text extraction", file=sys.stderr)
                    items = extract_amounts_from_text(
                        client=client,
                        model_name=model_name,
                        page_text=page_data["text"],
                        page_number=page_num,
                        pdf_name=pdf_name
                    )
                else:
                    # Use vision-based extraction for image PDFs
                    print(f"Page {page_num}: Using vision extraction (image-based PDF)", file=sys.stderr)
                    base64_image = convert_page_to_image(pdf_path, page_num)
                    items = extract_amounts_from_image(
                        client=client,
                        base64_image=base64_image,
                        page_number=page_num,
                        pdf_name=pdf_name
                    )
                
                all_items.extend(items)
                
            except Exception as e:
                result["failed_pages"].append({
                    "page": page_num,
                    "reason": str(e)
                })
        
        # Update result with extracted items
        result["amounts"] = all_items
        result["total_items"] = len(all_items)
        
    except FileNotFoundError:
        result["status"] = "error"
        result["error"] = f"PDF file not found: {pdf_path}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python total_price_extraction.py <pdf_path>")
        print("Example: python total_price_extraction.py data/invoice.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    # Extract amounts
    result = extract_invoice_amounts(pdf_path)
    
    # Output as formatted JSON
    print(json.dumps(result, indent=2))
    
    # Save to output file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{Path(pdf_path).stem}_amounts.json"
    with open(output_file, "w") as f:
        json.dump(result, indent=2, fp=f)
    
    print(f"\nResults saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
