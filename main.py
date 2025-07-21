from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy
from scipy.spatial.distance import euclidean
import voyageai
from io import BytesIO
import requests

app = FastAPI(
    title="VoyageAI Image Vectorizer",
    description="Create dense vectors for images using VoyageAI's multimodal embedding model",
    version="1.0",
    contact={
        "name": "Pat Wendorf",
        "email": "pat.wendorf@mongodb.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/license/mit/",
    }
)

# Initialize VoyageAI client
vo = voyageai.Client()

def similarity(v1, v2):
    # Define two dense vectors as NumPy arrays
    vector1 = numpy.array(v1)
    vector2 = numpy.array(v2)

    # Compute Euclidean distance
    euclidean_distance = euclidean(vector1, vector2)

    # Compute dot product
    dot_product = numpy.dot(vector1, vector2)

    # Compute cosine similarity
    cosine_similarity = numpy.dot(vector1, vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))

    return {"euclidean": euclidean_distance, "dotProduct": dot_product, "cosine": cosine_similarity}

def get_voyage_image_embedding(image):
    result = vo.multimodal_embed([[image]], model="voyage-multimodal-3")
    return result.embeddings[0]

def get_voyage_text_embedding(text):
    result = vo.multimodal_embed([[text]], model="voyage-multimodal-3")
    return result.embeddings[0]

@app.post("/upload_image_vector")
async def upload_image_vector(image: UploadFile):
    image = Image.open(image.file)
    return get_voyage_image_embedding(image)

@app.post("/text_vector")
async def url_image_vector(text: str):
    return get_voyage_text_embedding(text)

@app.post("/url_image_vector")
async def url_image_vector(image_url: str):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return get_voyage_image_embedding(image)

@app.post("/upload_image_image_similarity")
async def upload_image_image_similarity(image1: UploadFile, image2: UploadFile):
    i1 = Image.open(image1.file)
    i2 = Image.open(image2.file)
    v1 = get_voyage_image_embedding(i1)
    v2 = get_voyage_image_embedding(i2)
    return similarity(v1, v2)

@app.post("/url_image_image_similarity")
async def url_image_image_similarity(image_url1: str, image_url2: str):
    r1 = requests.get(image_url1)
    r2 = requests.get(image_url2)
    i1 = Image.open(BytesIO(r1.content))
    i2 = Image.open(BytesIO(r2.content))
    v1 = get_voyage_image_embedding(i1)
    v2 = get_voyage_image_embedding(i2)
    return similarity(v1, v2)

@app.post("/upload_image_text_similarity")
async def upload_image_text_similarity(image: UploadFile, text: str):
    img = Image.open(image.file)
    v1 = get_voyage_image_embedding(img)
    v2 = get_voyage_text_embedding(text)
    return similarity(v1, v2)

@app.post("/url_image_text_similarity")
async def url_image_text_similarity(image_url: str, text: str):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    v1 = get_voyage_image_embedding(img)
    v2 = get_voyage_text_embedding(text)
    return similarity(v1, v2)
