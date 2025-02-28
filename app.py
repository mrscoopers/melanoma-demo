import configparser
import torch
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client import models
from collections import Counter
from PIL import Image
from ollama import generate
import streamlit as st
import requests
from io import BytesIO

##############################################
#                 Config Setup               #
##############################################
st.set_page_config(
    page_title="Medical Assistant for Dermatoscopic Images",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="https://storage.googleapis.com/demo-skin-cancer/qdrant-logo.png"
)

config = configparser.ConfigParser()
config.read("config.ini")

# Qdrant
qdrant_cloud_url = config["qdrant"]["cloud_url"]
qdrant_api = config["secrets"]["api_key"]
client = QdrantClient(url=qdrant_cloud_url, api_key=qdrant_api)
collection_name = "melanoma_main_collection" 

# Ollama
OLLAMA_MODEL = "deepseek-llm"

##############################################
#        Model and Processor        #
##############################################
@st.cache_resource
def load_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
    return device, processor, model

device, processor, model = load_model_and_processor()

##############################################
#     Embedding Extraction (ViT)             #
##############################################

def get_embeddings_query(image):
    """Extract cls_output, average-pooled patches, and raw patches from the image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
        outputs = outputs.cpu().numpy()

    patches = outputs[0, 1:, :]     # all but the CLS token
    cls_output = outputs[0, 0, :]   # the CLS token
    pooled_patches = patches.mean(axis=0)

    del outputs, inputs
    torch.cuda.empty_cache()

    return cls_output, pooled_patches, patches

##############################################
#            Qdrant Search Helpers           #
##############################################
def search_knn(image_embedding, k=10):
    """Search the Qdrant collection by a single 'CLS' vector."""
    try:
        result = client.query_points(
            collection_name="melanoma_main_collection",
            query=image_embedding,
            using="cls",
            limit=k,
            with_payload=True
        ).points
        return result
    except Exception as e:
        st.error("Error: Unable to retrieve data from the knowledge base. Please check your connection and try again.")
        return []

def search_rerank_knn(image_embedding, patch_embeddings, prefetch_k=20, k=10):
    """
    Example of a 2-stage retrieval:
      1) Prefetch with 'pooled_patches'
      2) Rerank with patch-level embeddings
    """
    try:
        result = client.query_points(
            collection_name="melanoma_main_collection",
            prefetch=models.Prefetch(
                query=image_embedding,
                using="pooled_patches",
                limit=prefetch_k
            ),
            query=patch_embeddings,
            using="patches",
            limit=k,
            with_payload=True
        ).points
        return result
    except Exception as e:
        st.error("Error: Unable to retrieve data from the knowledge base. Please check your connection and try again.")
        return []

##############################################
#           KNN Classification               #
##############################################

def KNN_classifier_dx_with_rerank(image_embedding, patch_embeddings, k=10):
    """
    2-stage KNN classification using rerank approach.
    """
    points = search_rerank_knn(image_embedding, patch_embeddings, k*2, k)
    if not points:
        return None, 0

    diagnoses = [p.payload.get("dx", "") for p in points]
    counter = Counter(diagnoses)
    two_most_common = counter.most_common(2)

    # If there's a tie, increase k until a maximum
    if len(two_most_common) > 1 and two_most_common[0][1] == two_most_common[1][1] and k <= 50:
        return KNN_classifier_dx_with_rerank(image_embedding, patch_embeddings, k + 5)

    dx_class = two_most_common[0][0]
    dx_confidence = two_most_common[0][1] / k
    return dx_class, dx_confidence

##############################################
#       Simple RAG via DeepSeek            #
##############################################
def rag_deepseek(dx_class, sureness):
    """
    Sends a simple prompt to the deepseek-llm
    """
    prompt = f"""
        You are an AI medical assistant designed to help medical professionals with skin lesion categorization. 
        You're providing information to assist them in the diagnosis of skin lesions.

        ## Context:
        - The user is a doctor who has uploaded an image of a skin lesion to classify.
        - The app uses vector search to find similar images from a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
        
        ## Task Context
        The classifier result for the uploaded image is:
        - Diagnosis: {dx_class}
        - Confidence Score: {sureness:.2%}

        ## Task:
        Generate a structured and informative piece of text that includes:
        1. A disclaimer emphasizing that this AI tool is only suitable for an assistence to a medical expert.
        2. A clear presentation of the diagnosis and confidence score.
        3. A brief but precise explanation of the diagnosed condition, ensuring it is medically relevant.

        ## Output Formatting:
        - Don't use personal tone, greetings or chatty manner.
        - The disclaimer must appear at the beginning of the response.
        - The diagnosis and confidence score must be clearly stated.
        - The response should be formatted in markdown for readability.
        - Use neutral, professional tone.

        ## Information about the uploaded image classification to a doctor:
    """

    try:
        chunks = generate(model=OLLAMA_MODEL, prompt=prompt, options={"seed": 42, "temperature":0}, stream=True)
        for chunk in chunks:
            yield chunk["response"]
    except Exception as e:
        yield f"AI assitant error"

##############################################
#             Streamlit Frontend             #
##############################################


st.title("Medical Assistant for Dermatoscopic Images")
st.markdown("### Upload an image to get an AI medical assistant diagnosis or retrieve similar cases from the database.")

# Initialize session state variables
if 'image' not in st.session_state:
    st.session_state.image = None
if 'diagnosis_text' not in st.session_state:
    st.session_state.diagnosis_text = None
if 'similar_images' not in st.session_state:    
    st.session_state.similar_images = None  

# Upload Section
# Create tabs for image selection methods
image_source = st.radio(
    "Select image source:",
    ["Upload Image", "Enter Image Public URL", "Choose Example Images"],
    horizontal=True
)

# OPTION 1: Upload image
if image_source == "Upload Image":
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if image_file:
        # Clear diagnosis when new image is uploaded
        st.session_state.diagnosis_text = None
        st.session_state.similar_images = None
        st.session_state.image = Image.open(image_file)

# OPTION 2: Image from URL
elif image_source == "Enter Image Public URL":
    image_url = st.text_input("Enter the public URL of an image:", "")
    if image_url:
        try:
            response = requests.get(image_url)
            # Clear diagnosis when new image is selected from URL
            st.session_state.diagnosis_text = None
            st.session_state.similar_images = None
            st.session_state.image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")

# OPTION 3: Example images
elif image_source == "Choose Example Images":
    st.markdown("### Select an example:")
    
    # Example images with their corresponding diagnoses
    example_images = [
        {"url": "https://storage.googleapis.com/demo-skin-cancer/test_ISIC_0024431_HAM_0002134.jpeg", 
         "diagnosis": "Basal cell carcinoma"},
        {"url": "https://storage.googleapis.com/demo-skin-cancer/test_ISIC_0031918_HAM_0003141.jpeg", 
         "diagnosis": "Actinic keratosis"},
        {"url": "https://storage.googleapis.com/demo-skin-cancer/test_ISIC_0025901_HAM_0005562.jpeg", 
         "diagnosis": "Melanocytic nevus"},
    ]
    
    # Cache example images to load them only once
    @st.cache_data
    def load_example_image(url):
        try:
            response = requests.get(url)
            return Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Error loading example image: {e}")
            return None
    
    # Display example images in a row
    cols = st.columns(len(example_images))
    
    for i, (col, img_data) in enumerate(zip(cols, example_images)):
        with col:
            example_img = load_example_image(img_data["url"])
            if example_img:
                st.image(example_img, width=400)
                st.write(f"**Diagnosis:** {img_data['diagnosis']}")
                if st.button(f"Select Image {i+1}", key=f"example_{i}"):
                    # Clear diagnosis when new example image is selected
                    st.session_state.similar_images = None
                    st.session_state.diagnosis_text = None
                    st.session_state.image = example_img

# If an image is selected (from any source), proceed with analysis
if st.session_state.image:
    # Get embeddings from the image
    with st.spinner("Extracting image features..."):
        cls_output, pooled_patches, patches = get_embeddings_query(st.session_state.image)
    
    st.markdown("---")
    # Show the last selected image
    st.subheader("Selected Image")
    st.image(st.session_state.image, width=400)
    
    # ---------------- Diagnosis Section ----------------
    st.subheader("AI Assistant's Diagnosis:")
    st.markdown(
        "Click **Run Diagnosis** to get a skin lesion diagnosis from the AI assistant:"
    )
    
    # Two-column layout: left for button, right for streaming output
    col_diag_button, col_diag_output = st.columns([1, 2])
    with col_diag_button:
        run_diag = st.button("Run Diagnosis", key="diag_button")
        st.image("https://storage.googleapis.com/demo-skin-cancer/DeepSeek_RAG_logo.png", width=300)
    with col_diag_output:
        diag_output = st.empty()  # Reserved placeholder for diagnosis result
        
        # Display existing diagnosis if available
        if st.session_state.diagnosis_text:
            diag_output.markdown(st.session_state.diagnosis_text)
    
        if run_diag:
            diag_output.empty()
            with st.spinner("Processing image for diagnosis..."):
                melanoma_class, sureness = KNN_classifier_dx_with_rerank(pooled_patches, patches)
            if melanoma_class:
                diag_text = ""
                # Stream diagnosis text to the fixed output container
                with st.spinner("Generating diagnosis..."):
                    for chunk_text in rag_deepseek(melanoma_class, sureness):
                        diag_text += chunk_text
                        diag_output.markdown(diag_text)
                # Store the generated diagnosis text in session state
                st.session_state.diagnosis_text = diag_text
            else:
                st.error("Classification error. Please try again.")
                if st.button("Retry"):
                    st.rerun()
    
    st.markdown("---")
    
    # ---------------- Similar Cases Section ----------------
    st.subheader("Similar Cases")
    st.markdown(
        "Click **Show Top-5 Similar Images** to view cases from the database that are similar to the selected image."
    )
    
    def display_images():
        main_col, legend_col = st.columns([2, 1])
                
        with main_col:
            st.subheader("Retrieved Information:")
            for point in st.session_state.similar_images:
                with st.container():
                    image_col, info_col = st.columns([1, 1])
                    with image_col:
                        st.image(point.payload["url"], width=400)
                    with info_col:
                        st.markdown(f"""
                        | **Attribute**   | **Value** |
                        |----------------|-----------|
                        | **Age:**       | {int(point.payload.get('age', 'N/A'))} |
                        | **Sex:**       | {point.payload.get('sex', 'N/A')} |
                        | **Diagnosis:** | {point.payload.get('dx', 'N/A')} |
                        | **Type:**      | {point.payload.get('dx_type', 'N/A')} |
                        | **Localization:** | {point.payload.get('localization', 'N/A')} |
                        """)
            
        # Legend column on the right
        with legend_col:
            st.subheader("Legend")
            st.markdown("""
                #####            
                ##### Diagnosis Confirmation Types:
                - **histo**: Histopathology confirmed
                - **follow_up**: Follow-up confirmed
                - **consensus**: Expert consensus
                - **confocal**: Confocal microscopy confirmed
                
                #####
                ##### Localizations:
                - **face**: Face area
                - **trunk**: Torso area
                - **scalp**: Head/scalp area
                - **acral**: Hands/feet
                - **back**: Back area
                - **abdomen**: Stomach area
                - **chest**: Chest area
                - **upper extremity**: Arms area
                - **lower extremity**: Legs area
                - **neck**: Neck area
                - **genital**: Genital area
                - **ear**: Ear area
                - **foot**: Foot area
                - **hand**: Hand area
                - **unknown**: Unknown area
                """)
    run_top_similar = st.button("Show Top-5 Similar Images", key="similar_button")

    if st.session_state.similar_images and not run_top_similar:
        display_images()

    if run_top_similar:
        # Clear previously displayed images when button is pressed
        st.session_state.similar_images = None
        
        with st.spinner("Searching for similar cases..."):
            similar_images = search_knn(cls_output, k=5)
            if similar_images:
                st.session_state.similar_images = similar_images
                display_images()
            else:
                st.error("Searching error. Please try again.")
                if st.button("Retry"):
                    st.rerun()
else:
    st.info("Please select an image using one of the methods above to begin.")