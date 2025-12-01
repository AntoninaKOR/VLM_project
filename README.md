There is web interface (gradio) for pretrained VLM for image description and OCR.
# Requirements
You should have cuda >= 12.1.
Be sure you have installed nvidia container toolkit.
# Instruction
Ways to run the project:
1. Build and run:
```
docker-compose up -d --build    
```     

2. You can also use `docker-compose.hf.yml` to mount your Hugging Face cache directories for faster model loading.
```     
docker compose -f docker-compose.yml -f docker-compose.hf.yml up -d
```     
3. Build:
```
docker compose build
```
Then run:
```
    docker run --name transformers-container -p 7860:7860 \
  -v "$(pwd)/hf_cache:/app/hf_cache" \
  -e HF_HOME=/app/hf_cache \
  -e TRANSFORMERS_CACHE=/app/hf_cache/transformers \
  my-image
```
4. Pull image 
``` 
docker pull antoninakar/gradio_interface_smolvlm:latest
```
Then bild and run by any way above.
# Interface
Choose device in setting firstly. You can't change it without restarting container.
Access web interface:       
Default access web interface at `http://localhost:7860` or `http://localhost:your_port` if you change the port when you have run container
You can save model outputs to `./data` folder on host machine.
