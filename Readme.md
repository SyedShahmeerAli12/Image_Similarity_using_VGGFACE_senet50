
title: Image Similarity Using VGGFACE Senet50
emoji: 📉
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit





# Project Overview

This project utilizes the VGGFace model with SENet50 architecture to predict the similarity of uploaded images to various Hollywood and Bollywood actors.

## Setup Instructions

### Step 1: Create a Virtual Environment

First, create a virtual environment for this project. You can do this using Python's `venv` module. Replace `venv_name` with your preferred name for the virtual environment.

### Step 2: Activate the Virtual Environment

Activate the virtual environment based on your operating system.

- **Windows:**

  ```bash
  .\venv_name\Scripts\activate
  
### macOS/Linux:
source venv_name/bin/activate

### Step 3: Install Dependencies
Install the required Python packages listed in requirements.txt. This can be done using pip, Python's package installer.

pip install -r requirements.txt


### Step 4: Run test.py (Optional)
You can optionally run test.py to verify that everything is set up correctly.

python test.py

### Step 5: Run the Streamlit App
To start the Streamlit application, use the following command. This will launch the application and open it in your default web browser.

streamlit run app.py
