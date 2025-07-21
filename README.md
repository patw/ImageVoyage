# ImageVoyage 🚀📸

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ImageVoyage is a FastAPI service that generates dense vector embeddings for images using VoyageAI's multimodal embedding model, and calculates similarity between images.

## Features

- Generate vector embeddings for images from:
  - File uploads
  - Public URLs
- Calculate similarity between two images using:
  - Euclidean distance
  - Dot product
  - Cosine similarity

## Quick Start

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended for dependency management)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/patw/ImageVoyage.git
cd ImageVoyage
```

2. Install dependencies:
```bash
poetry install
```

3. Set your VoyageAI API key as an environment variable:
```bash
export VOYAGE_API_KEY="your-api-key-here"
```

### Running the Service

Start the FastAPI server:
```bash
poetry run uvicorn main:app --reload
```

The API documentation will be available at: http://localhost:8000/docs

## API Endpoints

- `POST /upload_image_vector` - Generate embedding from uploaded image file
- `POST /url_image_vector` - Generate embedding from image URL
- `POST /upload_image_image_similarity` - Compare two uploaded images
- `POST /url_image_image_similarity` - Compare two images from URLs

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
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

👤 **Pat Wendorf**
- GitHub: [@patw](https://github.com/patw)
- Email: pat.wendorf@mongodb.com
