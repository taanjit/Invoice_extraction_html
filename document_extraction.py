"""
PDF Invoice Content Extractor
Extracts content from PDF invoice files and saves as both HTML and JSON.

Output formats:
- HTML: Styled table view
- JSON: Structured data with Description, Total_Price_Text, Quantity_Text, Unit_Text, Unit_Price_Label

Supports both text-based and image-based (scanned) PDFs using OpenAI Vision API.
"""

import os
import io
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
import yaml
from openai import OpenAI


def load_model_config():
    """Load model configuration from YAML file."""
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model', {}).get('name', 'gpt-4o')


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    env_path = Path(__file__).parent / ".env"
    with open(env_path, 'r') as f:
        content = f.read().strip()
    
    if ":" in content:
        api_key = content.split(":", 1)[1].strip()
    else:
        api_key = content.strip()
    
    return api_key


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    doc.close()
    return text


def get_pdf_page_count(pdf_path: str) -> int:
    """Get the total number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def extract_images_from_pdf(pdf_path: str, dpi: int = 300) -> list:
    """Extract pages as high-resolution images from a PDF file."""
    doc = fitz.open(pdf_path)
    images = []
    
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        images.append({
            'base64': img_base64,
            'page_num': page_num + 1
        })
    
    doc.close()
    return images


def extract_structured_data_from_text(client: OpenAI, text: str, model_name: str, pdf_name: str) -> tuple:
    """Use OpenAI to extract structured data from the PDF text."""
    
    prompt = """You are analyzing an INVOICE document. Extract ALL line items from this invoice.

For EACH line item, extract:
- Description: Full item description or product name
- Total_Price_Text: The EXACT total price value shown in the document for this line item
- Quantity_Text: Quantity ordered (numeric value)
- Unit_Text: Unit of measurement (e.g., KG, BTL, PCS, BOX, PKT)
- Unit_Price_Label: Price per unit (numeric value)

CRITICAL RULES:
- Total_Price_Text MUST be read DIRECTLY from the document as-is
- DO NOT calculate Total_Price_Text from Quantity × Unit Price
- If Total_Price_Text is not visible in the document, use 0 or leave empty
- Only extract values that are explicitly printed in the document
- Use numeric values for prices and quantities (not strings)
- Return as JSON: {"items": [...]}

Invoice text:
"""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an expert invoice parser. Extract all line items from invoices accurately. Always return valid JSON with numeric values for prices and quantities."
            },
            {
                "role": "user",
                "content": prompt + text
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    items = parse_json_response(response.choices[0].message.content, pdf_name, 1)
    return items, []


def extract_structured_data_from_images(client: OpenAI, images: list, model_name: str, pdf_name: str, debug: bool = True) -> tuple:
    """Use OpenAI Vision API to extract structured data from PDF images."""
    
    all_items = []
    failed_pages = []
    
    for img_data in images:
        page_num = img_data['page_num']
        img_base64 = img_data['base64']
        
        if debug:
            print(f"    Processing page {page_num}...")
        
        content = [
            {
                "type": "text",
                "text": """You are analyzing an INVOICE document image. Extract ALL line items visible on this page.

For EACH line item found, extract:
- Description: Full item description, product name, or service description
- Total_Price_Text: The EXACT total price value shown in the document for this line item (numeric only)
- Quantity_Text: Quantity (numeric value only)
- Unit_Text: Unit of measurement (e.g., KG, BTL, PCS, BOX, PKT, LTR)
- Unit_Price_Label: Price per unit (numeric value only, no currency symbols)

CRITICAL RULES:
1. Total_Price_Text MUST be read DIRECTLY from the document as printed
2. DO NOT calculate Total_Price_Text from Quantity × Unit Price
3. If the total price column is not visible, use 0
4. Only extract values that are explicitly shown in the document
5. Look for columns like "Total", "Amount", "Line Total", "Ext. Price"
6. Use numeric values for all price and quantity fields (floats/integers, not strings)
7. For Unit_Text, use common abbreviations: KG, BTL, PCS, BOX, PKT, LTR, CTN, etc.

Return ONLY a JSON object: {"items": [{"Description": "", "Total_Price_Text": 0.0, "Quantity_Text": 0, "Unit_Text": "", "Unit_Price_Label": 0.0}, ...]}

If no invoice line items are found, return: {"items": []}"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "high"
                }
            }
        ]
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert invoice and document parser. Extract structured data from invoice images. Return numeric values for prices and quantities. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=4096
            )
            
            response_content = response.choices[0].message.content
            page_items = parse_json_response(response_content, pdf_name, page_num)
            
            if debug:
                print(f"      Found {len(page_items)} items on page {page_num}")
            
            all_items.extend(page_items)
            
        except Exception as e:
            print(f"      Error processing page {page_num}: {str(e)}")
            failed_pages.append(page_num)
            continue
    
    return all_items, failed_pages


def parse_json_response(content: str, pdf_name: str, page_num: int) -> list:
    """Parse JSON response from OpenAI and format items with metadata."""
    try:
        result = json.loads(content)
        
        raw_items = []
        if isinstance(result, list):
            raw_items = result
        elif isinstance(result, dict):
            for key in ["items", "data", "line_items", "records", "invoice_items", "lines"]:
                if key in result and isinstance(result[key], list):
                    raw_items = result[key]
                    break
        
        # Format items with required structure
        formatted_items = []
        for item in raw_items:
            formatted_item = {
                "Description": str(item.get("Description", item.get("description", ""))),
                "Total_Price_Text": convert_to_number(item.get("Total_Price_Text", item.get("total_price", 0))),
                "Quantity_Text": convert_to_number(item.get("Quantity_Text", item.get("qty", item.get("quantity", 0)))),
                "Unit_Text": str(item.get("Unit_Text", item.get("unit", ""))),
                "Unit_Price_Label": convert_to_number(item.get("Unit_Price_Label", item.get("unit_price", 0))),
                "_confidence": "high",
                "_flags": [],
                "_page": page_num,
                "_pdf": pdf_name
            }
            formatted_items.append(formatted_item)
        
        return formatted_items
        
    except json.JSONDecodeError as e:
        print(f"      JSON Parse Error: {str(e)}")
        return []


def convert_to_number(value):
    """Convert a value to a number (int or float)."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
        try:
            if "." in cleaned:
                return float(cleaned)
            return int(cleaned)
        except ValueError:
            return 0
    return 0


def save_json_output(data: dict, output_path: str):
    """Save extraction results as JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  JSON file saved to: {output_path}")


def generate_html(data: list, output_path: str, pdf_name: str = ""):
    """Generate HTML file with the extracted data in a styled table."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Invoice Extraction Results{f' - {pdf_name}' if pdf_name else ''}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); min-height: 100vh; padding: 40px 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #e94560; margin-bottom: 10px; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }}
        .subtitle {{ text-align: center; color: rgba(255, 255, 255, 0.7); margin-bottom: 30px; font-size: 1rem; }}
        .table-container {{ background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 30px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
        thead {{ background: linear-gradient(135deg, #e94560 0%, #c23a51 100%); }}
        th {{ padding: 18px 15px; text-align: left; color: white; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border: none; }}
        th:first-child {{ border-radius: 10px 0 0 0; }}
        th:last-child {{ border-radius: 0 10px 0 0; }}
        tbody tr {{ transition: all 0.3s ease; border-bottom: 1px solid #e0e0e0; }}
        tbody tr:hover {{ background: linear-gradient(135deg, #fff5f7 0%, #ffe8ec 100%); transform: scale(1.01); }}
        tbody tr:nth-child(even) {{ background-color: #f8f9fa; }}
        td {{ padding: 15px; color: #333; vertical-align: top; }}
        .desc-col {{ max-width: 400px; line-height: 1.5; }}
        .num-col {{ text-align: right; font-family: 'Courier New', monospace; font-weight: 500; }}
        .unit-col {{ text-align: center; }}
        .page-col {{ text-align: center; color: #666; font-size: 0.85rem; }}
        .summary {{ margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px; text-align: center; }}
        .footer {{ text-align: center; margin-top: 30px; color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; }}
        .no-data {{ text-align: center; padding: 40px; color: #666; font-size: 1.2rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 Invoice Extraction Results</h1>
        <p class="subtitle">{pdf_name if pdf_name else 'Extracted Data'}</p>
        <div class="table-container">
"""
    
    if data:
        html_content += """            <table>
                <thead>
                    <tr>
                        <th>Description</th>
                        <th>Qty</th>
                        <th>Unit</th>
                        <th>Unit Price</th>
                        <th>Total Price</th>
                        <th>Page</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for item in data:
            desc = str(item.get("Description", "")).replace("<", "&lt;").replace(">", "&gt;")
            qty = item.get("Quantity_Text", "")
            unit = str(item.get("Unit_Text", "")).replace("<", "&lt;").replace(">", "&gt;")
            unit_price = item.get("Unit_Price_Label", "")
            total_price = item.get("Total_Price_Text", "")
            page = item.get("_page", "")
            
            html_content += f"""                    <tr>
                        <td class="desc-col">{desc}</td>
                        <td class="num-col">{qty}</td>
                        <td class="unit-col">{unit}</td>
                        <td class="num-col">{unit_price}</td>
                        <td class="num-col">{total_price}</td>
                        <td class="page-col">{page}</td>
                    </tr>
"""
        
        html_content += f"""                </tbody>
            </table>
            <div class="summary">Total Items: {len(data)}</div>
"""
    else:
        html_content += """            <div class="no-data"><p>No structured data could be extracted.</p></div>
"""
    
    html_content += """        </div>
        <div class="footer"><p>Generated using OpenAI Vision API | Invoice Content Extractor</p></div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  HTML file saved to: {output_path}")


def main():
    """Main function to orchestrate the extraction process."""
    project_dir = Path(__file__).parent
    
    # Load API key and model config
    api_key = load_environment()
    model_name = load_model_config()
    client = OpenAI(api_key=api_key)
    
    print(f"Using model: {model_name}")
    print(f"Image rendering: 300 DPI (high resolution)")
    
    # Find PDF files in the data directory
    data_dir = project_dir / "data"
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the data directory.")
        return
    
    # Process each PDF file
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing INVOICE: {pdf_path.name}")
        print('='*60)
        
        pdf_name = pdf_path.stem
        num_pages = get_pdf_page_count(str(pdf_path))
        
        print(f"  Total pages: {num_pages}")
        print("  Extracting text from PDF...")
        text = extract_text_from_pdf(str(pdf_path))
        
        if text.strip() and len(text.strip()) > 100:
            print(f"  Found {len(text)} characters of text. Using text extraction...")
            items, failed_pages = extract_structured_data_from_text(client, text, model_name, pdf_name)
        else:
            print("  PDF appears to be image-based. Using Vision API for OCR...")
            images = extract_images_from_pdf(str(pdf_path), dpi=300)
            print(f"  Extracted {len(images)} page(s) at 300 DPI")
            print(f"  Sending to {model_name} for analysis...")
            items, failed_pages = extract_structured_data_from_images(client, images, model_name, pdf_name, debug=True)
        
        print(f"\n  Total items extracted: {len(items)}")
        
        # Save JSON output
        json_output = {
            "status": "success" if items else "no_data",
            "items": items,
            "num_pages": num_pages,
            "failed_pages": failed_pages
        }
        json_path = project_dir / f"{pdf_name}_output.json"
        save_json_output(json_output, str(json_path))
        
        # Save HTML output
        html_path = project_dir / f"{pdf_name}_extracted.html"
        generate_html(items, str(html_path), pdf_path.name)


if __name__ == "__main__":
    main()
