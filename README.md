# PDF Invoice Content Extractor

A Python-based tool that extracts structured data from PDF invoice documents using OpenAI's Vision API. The tool supports both text-based and image-based (scanned) PDFs.

## Features

- **Multi-format PDF Support**: Handles both text-based PDFs and scanned/image-based invoices
- **High-Resolution OCR**: Renders PDFs at 300 DPI for accurate text extraction
- **AI-Powered Extraction**: Uses OpenAI GPT models to intelligently parse invoice data
- **Dual Output Formats**: Generates both JSON and HTML outputs
- **Data Validation**: Validates extracted values to ensure accuracy (rejects calculated values)
- **Configurable Model**: Model selection via YAML configuration file

## Output Structure

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
      "_pdf": "invoice_filename"
    }
  ],
  "num_pages": 6,
  "failed_pages": []
}
```

### Fields Extracted
| Field | Description |
|-------|-------------|
| `Description` | Product/item name |
| `Total_Price_Text` | Line total (read directly from document) |
| `Quantity_Text` | Quantity ordered |
| `Unit_Text` | Unit of measurement (KG, BTL, PCS, etc.) |
| `Unit_Price_Label` | Price per unit |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Variables
Create a `.env` file with your OpenAI API key:
```
OpenAi key : sk-your-api-key-here
```

### Model Configuration
Edit `config/model_config.yaml` to select the AI model:
```yaml
model:
  name: "gpt-4.1-mini"
```

## Usage

1. Place PDF files in the `data/` directory
2. Run the extraction script:
```bash
python document_extraction.py
```
3. Find outputs in the project root:
   - `{filename}_output.json` - Structured JSON data
   - `{filename}_extracted.html` - Styled HTML table view

## Project Structure

```
cms_document_reader/
├── config/
│   └── model_config.yaml    # Model configuration
├── data/                    # Input PDF files
├── document_extraction.py   # Main extraction script
├── requirements.txt         # Python dependencies
├── .env                     # API key (not in version control)
└── README.md
```

## Dependencies

- `openai` - OpenAI API client
- `PyMuPDF` (fitz) - PDF processing and rendering
- `python-dotenv` - Environment variable management
- `pyyaml` - YAML configuration parsing

## Data Validation

The tool includes validation to ensure `Total_Price_Text` is read directly from the document:
- Values matching `Quantity × Unit_Price` are rejected as potentially calculated
- Missing values are represented as `[]`
- Only explicitly printed values are extracted

## License

MIT License
