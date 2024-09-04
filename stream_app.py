pip install --upgrade accelerate transformers
import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Load the pre-trained model and processor
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    load_in_4bit=True,
    _attn_implementation='eager'
)

processor = AutoProcessor.from_pretrained(model_id,
                                          trust_remote_code=True,
                                          num_crops=4
                                         )

# Create a Streamlit application
st.title("Instagram Caption Generator📝")
st.write("Upload an image and get captions for Instagram posts!")

# Create an uploader for the image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.write("Uploaded Image:")
    st.image(uploaded_image, width=400)

    # Open the image using PIL
    image = Image.open(uploaded_image)

    # Create a placeholder for the image
    placeholder = "<|image_1|>"

    # Create a message with the placeholder
    messages = [
        {"role": "user", "content": placeholder + "generate 5 captions based on the provided image for instagram post remember each caption should be a medium length sentence and use emojis"},
    ]

    # Process the message and image
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

    # Generate the description
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 1,
        "do_sample": False,
    }

    generate_ids = model.generate(**inputs,
                                  eos_token_id=processor.tokenizer.eos_token_id,
                                  **generation_args
                                 )

    # Remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    # Display the description
    st.write("Captions:")
    st.write(response)
