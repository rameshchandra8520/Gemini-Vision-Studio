## Gemini Annotator

### **Project Description**:  
This project is an AI-powered object detection and visualization tool built using Streamlit and Google Gemini 2.0 API. It processes uploaded images, identifies objects using a large language model, and generates 2D bounding boxes with descriptive labels. Key features include:  
- Interactive image upload and prompt input through a user-friendly web interface.  
- AI-based object recognition with unique labeling of multiple objects.  
- Visualization of labeled bounding boxes directly on the uploaded images.  
- Real-time response generation leveraging Google Gemini's generative capabilities.  
- Integration with Python libraries like PIL for image rendering and manipulation.  



### Resources
- [Google Gemini 2.0](https://ai.google.dev/gemini-api/docs/models/gemini-v2#:~:text=Gemini%202.0%20Flash%20is%20now,streaming%20applications%20with%20tool%20use.)
- [Google GenAI SDK](https://pypi.org/project/google-genai/)
- [Streamlit](https://streamlit.io/)
- [Gemini API Key](https://aistudio.google.com/app/apikey)



### Additional Implementations
- **Offline Model Support**
    - Use open-source models (e.g., YOLO, CLIP, or LLaMA) for offline inference to reduce API dependency and costs.
- **Prompt Optimization**
    - Implement a prompt optimization algorithm to generate more accurate and relevant responses.
    - Introduce custom prompt templates based on the user's input type or goal.
- **More Detailed text responses**
    - Provide detailed descriptions of the detected objects and their relationships in the image.