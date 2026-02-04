# PDF Invoice Content Extractor

A Python tool that extracts structured data from PDF invoices using OpenAI's Vision API and saves the results as both JSON and HTML formats.

## Features

- **PDF Processing**: Handles both text-based and image-based (scanned) PDFs
- **OCR with Vision API**: Uses OpenAI GPT-4o for accurate text extraction from images
- **High Resolution**: Renders PDFs at 300 DPI for optimal OCR quality
- **Dual Output**: Generates both JSON and HTML output files
- **Invoice-Specific**: Optimized prompts for extracting invoice line items

## Output Format

### JSON Output
```json
{
  "status": "success",
  "items": [
    {
      "Description": "Coffee Nescafe Gold",
      "Total_Price_Text": 60.6,
      "Quantity_Text": 12,
      "Unit_Text": "BTL",
      "Unit_Price_Label": 5.05,
      "_confidence": "high",
      "_flags": [],
      "_page": 5,
      "_pdf": "invoice_name"
    }
  ],
  "num_pages": 6,
  "failed_pages": []
}
```

### HTML Output
Styled table view with all extracted invoice items.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API key** in `.env`:
   ```
   OpenAi key : sk-your-api-key-here
   ```

3. **Configure model** in `config/model_config.yaml`:
   ```yaml
   model:
     name: "gpt-4o-mini"
   ```

## Usage

1. Place PDF invoices in the `data/` directory
2. Run the extraction:
   ```bash
   python document_extraction.py
   ```
3. Output files are saved in the project root:
   - `{pdf_name}_output.json` - Structured JSON data
   - `{pdf_name}_extracted.html` - Styled HTML table

## Project Structure

```
cms_document_reader/
├── config/
│   └── model_config.yaml    # Model configuration
├── data/                    # Input PDF files
├── document_extraction.py   # Main extraction script
├── requirements.txt         # Python dependencies
├── .env                     # API key configuration
└── README.md
```

## Requirements

- Python 3.8+
- OpenAI API key with access to GPT-4o or GPT-4o-mini
