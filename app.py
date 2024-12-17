import json
import os
import random
import re
import tempfile

from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageColor, ImageDraw, ImageFont

load_dotenv()

def call_llm(img: Image, prompt: str) -> str:
    system_prompt = """
    You are an image analysis assistant. Analyze the image based on the user prompt and provide responses in a structured JSON format.

    ### General Guidelines:
    1. Always return a JSON object. Never include code fences or unstructured text.
    2. Limit to 25 objects, prioritizing objects relevant to the user's prompt.
    3. Be descriptive, concise, and precise when labeling and explaining objects.

    ### JSON Response Structure:
    Return a JSON object with the following keys:
    - "objects": A list of detected objects. Each object contains:
        - "box_2d": A list of coordinates [y1, x1, y2, x2] (normalized from 0 to 1000) for the object's bounding box.
        - "label": A descriptive name for the object, including unique characteristics (e.g., color, size, position).
        - "description": A detailed explanation or observation about the object (e.g., color, texture, material, or reflection details).
    - "extra_info": Additional information based on the user’s query:
        - If the query is about differences between two images, describe key visual differences.
        - If the query asks for an object’s color, list all objects that match the specified color.
        - If the query is about reflections, identify objects that appear reflective and explain why.
        - If the query asks for scene explanations, summarize what is happening in the image, including spatial relationships and prominent objects.

    ### Examples:

    #### Example 1: General Object Detection
    User Prompt: "What objects are in the image?"
    Response:
    {
    "objects": [
        {
        "box_2d": [195, 483, 479, 527],
        "label": "person in blue shirt",
        "description": "A tall person wearing a blue shirt and dark pants, standing near the center."
        },
        {
        "box_2d": [342, 150, 550, 300],
        "label": "red car",
        "description": "A small red car parked on the left side."
        }
    ],
    "extra_info": "The image shows a person and a red car in a daytime outdoor setting."
    }

    #### Example 2: Color-Based Query
    User Prompt: "What are the objects which are in color blue?"
    Response:
    {
    "objects": [
        {
        "box_2d": [195, 483, 479, 527],
        "label": "person in blue shirt",
        "description": "A tall person wearing a blue shirt and dark pants, standing near the center."
        }
    ],
    "extra_info": "The only blue object in the image is the person's shirt."
    }

    #### Example 3: Reflective Objects
    User Prompt: "How many reflecting objects are there?"
    Response:
    {
    "objects": [
        {
        "box_2d": [120, 430, 300, 500],
        "label": "glass table",
        "description": "A shiny glass table reflecting light in the center."
        },
        {
        "box_2d": [400, 200, 480, 250],
        "label": "mirror",
        "description": "A wall-mounted mirror reflecting parts of the room."
        }
    ],
    "extra_info": "There are 2 reflective objects: a glass table and a mirror."
    }

    #### Example 4: Image Differences
    User Prompt: "Find the difference between two images."
    Response:
    {
    "objects": [],
    "extra_info": "The two images differ in the following ways: 1) A red car is present in the first image but absent in the second. 2) A person in a blue shirt appears closer to the camera in the second image."
    }
    """


    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            safety_settings=[  # https://ai.google.dev/api/generate-content#v1beta.HarmCategory
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ],
        ),
    )
    print("Response from LLM", response.text)
    return response.text


def parse_json(json_input: str) -> str:
    match = re.search(r"```json\n(.*?)```", json_input, re.DOTALL)
    json_input = match.group(1) if match else ""
    return json_input


def plot_bounding_boxes(img: Image, bounding_boxes: str) -> Image:
    width, height = img.size
    colors = [colorname for colorname in ImageColor.colormap.keys()]
    draw = ImageDraw.Draw(img)

    bounding_boxes = parse_json(bounding_boxes)
    print("Bounding Boxes", bounding_boxes)

    bounding_boxes = json.loads(bounding_boxes)
    objects_boxes = bounding_boxes.get("objects", [])
    print("Objects Boxes", objects_boxes)

    for box in objects_boxes:
        color = random.choice(colors)
        print("bounding_box", box)
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(box["box_2d"][0] / 1000 * height)
        abs_x1 = int(box["box_2d"][1] / 1000 * width)
        abs_y2 = int(box["box_2d"][2] / 1000 * height)
        abs_x2 = int(box["box_2d"][3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        print(
            f"Absolute Co-ordinates: {box['label']}, {abs_y1}, {abs_x1},{abs_y2}, {abs_x2}",
        )

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw label
        draw.text(
            (abs_x1 + 8, abs_y1 + 6),
            box["label"],
            fill=color,
            font=ImageFont.truetype(
                # "Arial.ttf",
                # "path/to/your/font.ttf",
                "C:/Windows/Fonts/arial.ttf",
                size=14,
            ),
        )

    return img, bounding_boxes["extra_info"]


if __name__ == "__main__":
    st.set_page_config(page_title="Gemini Vision Studio", layout="wide")
    st.header("⚡️ Gemini Vision Studio")
    prompt = st.text_input("Enter your prompt")
    run = st.button("Run!")

    with st.sidebar:
        uploaded_image = st.file_uploader(
            accept_multiple_files=False,
            label="Upload your photo here:",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_image:
            with st.expander("View the image"):
                st.image(uploaded_image)

    if uploaded_image and run and prompt:
        temp_file = tempfile.NamedTemporaryFile(
            "wb", suffix=f".{uploaded_image.type.split('/')[1]}", delete=False
        )
        temp_file.write(uploaded_image.getbuffer())
        image_path = temp_file.name
        temp_file.close()

        img = Image.open(image_path)
        width, height = img.size
        resized_image = img.resize(
            (1024, int(1024 * height / width)), Image.Resampling.LANCZOS
        )
        os.unlink(image_path)
        print(
            f"Image Original Size: {img.size} | Resized Image size: {resized_image.size}"
        )

        with st.spinner("Running..."):
            response = call_llm(resized_image, prompt)
            plotted_image, extra_info= plot_bounding_boxes(resized_image, response)
        st.image(plotted_image)
        st.subheader("Detailed Information")
        st.write(extra_info)    