from openai import OpenAI
import streamlit as st
import os
from detection_module import detect_vessel_dimensions_from_bytes_aircraft
import datetime
import io
from PIL import Image
client = OpenAI(api_key="")


st.set_page_config(page_title="Aircraft Detection Dashboard", layout="centered")
st.title(" Aircraft Detection from Satellite Imagery")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

confidence = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
selected_category = st.text_input("Category to detect (or leave as 'all')", "all")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    
    with st.spinner("Detecting aircraft Now"):
        results = detect_vessel_dimensions_from_bytes_aircraft(
            image_bytes=image_bytes,
            category_to_detect=selected_category,
            confidence_threshold=confidence,
            is_aircraft=True
        )

    if results:
        st.success(f" {len(results)} object(s) detected.")
        
        # Display image
        img = Image.open(io.BytesIO(results[0]["annotated_image"]))
        st.image(img, caption="Annotated Detection Image", use_column_width=True)

        

        
        st.subheader("Detection Details")
        for idx, det in enumerate(results):
            st.write(f"### Object {idx + 1}")
            st.json({
                "Category": det['detected_category'],
                "Confidence": det['confidence'],
                "Box": det['bounding_box'],
                "Timestamp": det['metadata']['processing_timestamp_utc']
            })

    else:
        st.warning("No objects detected.")
        st.markdown("###  Ask the Aircraft Assistant")

user_input = st.text_input("Ask something about the detected aircraft:", "")

if st.button("Ask LLM") and user_input:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a military aviation expert."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )

        # Extract and display the reply
        reply = response.choices[0].message.content
        st.markdown("**LLM Response:**")
        st.markdown(reply)

    except Exception as e:
        st.error(f" Error talking to LLM: {str(e)}")

