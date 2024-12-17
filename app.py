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
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "Arial.ttf")

def call_llm(img: Image, prompt: str) -> str:
    system_prompt = """
    You are an image analysis assistant. Analyze the provided image based on the user's prompt and return responses in a structured JSON format.

    ### General Guidelines:
    1. Always return a JSON object. Never include code fences or unstructured text.
    2. Limit to 25 objects, prioritizing objects relevant to the user's prompt.
    3. Be descriptive, concise, and precise when labeling and explaining objects.
    4. Generate more detailed responses with proper heading, information based on the user's prompt and the image content in the "extra_info" key.

    ### JSON Response Structure:
    Return a JSON object with the following keys:
    - "objects": A list of detected objects. Each object contains:
        - "box_2d": A list of coordinates [y1, x1, y2, x2] (normalized from 0 to 1000) for the object's bounding box.
        - "label": A descriptive name for the object, including unique characteristics (e.g., color, size, position).
        - "description": A detailed explanation or observation about the object (e.g., color, texture, material, or reflection details).
    - "extra_info": should be **key-value pair** and will provide relevant details based on the user's specific prompt, such as explanations, translations, or differences or UI code snippets in the proper format.
        - The keys should be concise subheadings summarizing parts of the response.
        - The values should contain the more clear detailed answers based on the user’s question and the image content.
    """


    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            safety_settings=[ 
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ],
        ),
    )
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

    bounding_boxes = json.loads(bounding_boxes)
    objects_boxes = bounding_boxes.get("objects", [])

    for box in objects_boxes:
        color = random.choice(colors)
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(box["box_2d"][0] / 1000 * height)
        abs_x1 = int(box["box_2d"][1] / 1000 * width)
        abs_y2 = int(box["box_2d"][2] / 1000 * height)
        abs_x2 = int(box["box_2d"][3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1


        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw label
        draw.text(
            (abs_x1 + 8, abs_y1 + 6),
            box["label"],
            fill=color,
            font=ImageFont.truetype(
                # "Arial.ttf",
                FONT_PATH,
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

        with st.spinner("Running..."):
            response = call_llm(resized_image, prompt)
            plotted_image, extra_info= plot_bounding_boxes(resized_image, response)
        st.image(plotted_image)
        st.subheader("Detailed Information")
        for key, value in extra_info.items():
            # if value is string
            if isinstance(value, str):
                st.subheader(key)
                st.markdown(value)
            else :
                # if value is dict
                st.subheader(key)
                for k, v in value.items():
                    st.markdown(f"**{k}**: {v}")