# ImageVoyage 🚀📸

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ImageVoyage is a FastAPI service that generates dense vector embeddings for images using VoyageAI's multimodal embedding model, and calculates similarity between images.

## Features

- Generate vector embeddings for:
  - Images from file uploads
  - Images from public URLs
  - Text strings
- Calculate similarity between:
  - Two images
  - Image and text
  Using:
  - Euclidean distance
  - Dot product
  - Cosine similarity

## Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (modern Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/patw/ImageVoyage.git
cd ImageVoyage
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies with uv:
```bash
uv pip install -r requirements.txt
```

3. Set your VoyageAI API key as an environment variable:
```bash
export VOYAGE_API_KEY="your-api-key-here"
```

### Running the Service

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API documentation will be available at: http://localhost:8000/docs

## API Endpoints

- `POST /upload_image_vector` - Generate embedding from uploaded image file
- `POST /url_image_vector` - Generate embedding from image URL
- `POST /text_vector` - Generate embedding from text string
- `POST /upload_image_image_similarity` - Compare two uploaded images
- `POST /url_image_image_similarity` - Compare two images from URLs
- `POST /upload_image_text_similarity` - Compare uploaded image with text
- `POST /url_image_text_similarity` - Compare URL image with text

## Example Usage

```python
import requests

# Get embedding from URL
response = requests.post(
    "http://localhost:8000/url_image_vector",
    json={"image_url": "https://example.com/image.jpg"}
)
print(response.json())

# Compare two images
response = requests.post(
    "http://localhost:8000/url_image_image_similarity",
    json={
        "image_url1": "https://example.com/image1.jpg",
        "image_url2": "https://example.com/image2.jpg"
    }
)
print(response.json())

# Compare image with text
response = requests.post(
    "http://localhost:8000/url_image_text_similarity",
    json={
        "image_url": "https://example.com/image.jpg",
        "text": "a sunset over the ocean"
    }
)
print(response.json())

# Get text embedding
response = requests.post(
    "http://localhost:8000/text_vector",
    json={"text": "a beautiful landscape"}
)
print(response.json())
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

👤 **Pat Wendorf**
- GitHub: [@patw](https://github.com/patw)
- Email: pat.wendorf@mongodb.com
