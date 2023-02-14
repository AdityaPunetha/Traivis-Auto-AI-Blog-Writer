from fastapi import FastAPI
from pexels import get_pexels_image

from model import Generator
import json


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/create_sections")
def create_sections(prompt):
    sections = app.generator.generate_blog_headings(prompt)
    # sections = [
    #     " Understanding the basics of blockchains",
    #     " Creating a blockchain wallet",
    #     " Using the blockchain wallet",
    #     " Building a transaction to pay someone",
    #     " Using the blockchain",
    # ]
    return sections


@app.get("/api/create_blog")
def create_blog(title, sections):
    images = get_pexels_image(title, amount=len(sections))
    output = []
    image_count = 0
    if images == "No results found":
        image_count = 0
    else:
        image_count = len(images)
    output.append(f"<h1>{title}</h1>\n")
    c = 0
    for section in sections:
        expanded_section = app.generator.generate_blog_section(title, section)
        if c < image_count:
            # output.append(f"<h2>{section}</h2>\n<p>{expanded_section}</p>\n\n")
            output.append(f"<img src={images[c][2]} alt={images[c][0]}>\n")
            c += 1
        output.append(f"<h2>{section}</h2>\n<p>{expanded_section}</p>\n\n")
    return output


@app.on_event("startup")
async def startup_event():
    app.generator = Generator("EleutherAI/gpt-neo-2.7B")
