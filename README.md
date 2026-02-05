# Invoice Data Extraction Tool

This tool extracts structured line-item data from invoice PDF documents using a combination of PDF parsing and Large Language Models (LLMs). It handles both digital (text-based) and scanned (image-based) PDFs.

## Features

- **Dual Extraction Engine**: 
  - **Text Mode**: Uses PyMuPDF for fast extraction from digital PDFs.
  - **Vision Mode**: Automatically switches to OpenAI Vision for scanned or image-based invoices.
- **Structured Data**: Extracts `line_number`, `description`, `amount`, `Quantity`, and `Unit_price`.
- **JSON Output**: Produces standardized JSON results for easy integration.
- **Multipage Support**: Processes all pages in a document while maintaining page-level metadata.

## Prerequisites

- Python 3.8+
- OpenAI API Key

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd cms_document_reader
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Create a `.env` file in the root directory with your OpenAI API key:
   ```env
   OpenAi key : sk-your-api-key-here
   ```
   *Note: Ensure the key is prefixed with `sk-`.*

## Usage

Run the extraction script by providing the path to an invoice PDF:

```bash
python total_price_extraction.py data/your_invoice.pdf
```

### Example
```bash
python total_price_extraction.py data/MTBOWCHEETAHDN2025112425112705251059-1.pdf
```

## Output Format

The tool saves a JSON file in the `output/` directory with the following structure:

```json
{
  "status": "success",
  "pdf_name": "filename",
  "num_pages": 1,
  "total_items": 45,
  "amounts": [
    {
      "line_number": 1,
      "description": "Biscuits Assorted Cookies",
      "amount": 12.12,
      "Quantity": 6.0,
      "Unit_price": 2.02,
      "_page": 1,
      "_pdf": "filename"
    }
  ],
  "failed_pages": []
}
```

## Project Structure

- `total_price_extraction.py`: Main execution script.
- `config/model_config.yaml`: Configuration for the OpenAI model to use.
- `data/`: Directory for input PDF documents.
- `output/`: Directory where extracted JSON files are saved.
- `requirements.txt`: Python package dependencies.
