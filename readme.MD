# Aircraft & Vessel Detection Dashboard

A YOLOv8-based system that:
- Detects aircraft types (F-16, F-22, Mirage, etc.) and vessels from satellite images
- Runs on a clean Streamlit dashboard
- Integrates with OpenAI LLMs for user queries about detections


- Custom-trained model with classes: `a10`, `passenger`, `mirage`, etc.
- LLM integration using GPT-4 via OpenAI API
- Real-time annotation with confidence scores



```bash
pip install -r requirements.txt
streamlit run dashboard.py
"# aircraft_detection_dashboard" 
