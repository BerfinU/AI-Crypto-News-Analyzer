# Installation & Setup Guide

This guide contains all the steps needed to set up and run the project from scratch on your computer.

## Prerequisites

- **Python** (version 3.9 or higher recommended)
- **pip** (Python package manager)

## Step 1: Installation

### 1. Get Project Files
Download/clone the project files to a folder on your computer.

```bash
git clone https://github.com/BerfinU/ai-crypto-news-analyzer.git
cd ai-crypto-news-analyzer
```

### 2. Create Virtual Environment
This isolates the project's libraries from other projects on your system. Open terminal in the project root directory and run:

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Required Packages
All libraries needed to run the main application are listed in this file:

```bash
pip install -r requirements.txt
```

### 5. Install Playwright Browsers
This step is **mandatory** for the scraper to work:

```bash
pip install playwright
playwright install
```

## Step 2: Configuration

### 1. Set Up API Key
Create a text file named `.env` in the project root directory. Paste your Gemini API key from Google AI Studio as follows:

```env
GOOGLE_API_KEY="AIzaSy...YOUR_API_KEY_HERE"
```

### 2. Configure General Settings
Open the `config.yaml` file in the `config/` folder with a text editor and fill in the following fields according to your setup:

#### Model Configuration:
- **`model -> classifier_path`**: You must write the full path to where you placed the trained classification model (`crypto_model_finetuned` folder) that was provided to you.

**Example:**
```yaml
model:
  classifier_path: "C:/Users/Berfin/Desktop/Project/crypto_model_finetuned"
```

#### Telegram Configuration:
```yaml
telegram:
  bot_token: "YOUR_BOT_TOKEN_FROM_BOTFATHER"
  chat_id: "YOUR_CHANNEL_OR_GROUP_ID"  # Usually a negative number
```

## Step 3: Running the Application

### 1. Create Database (First Time Only)
If you're running the project for the first time, run the following command **only once** to create the `news.db` file and necessary tables:

```bash
python create_db.py
```

### 2. Start Main System (Scheduler)
This command starts the data collection and processing loop. You must keep this terminal open:

```bash
python scheduler.py
```

You will see logs starting to flow on the screen. This means the system is running.

### 3. View Dashboard
- Open a **new** terminal window
- Activate the virtual environment again (`venv\Scripts\activate`)
- Enter the following command to start the Streamlit interface:

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501` where you can view the dashboard.

## Quick Start Commands

After initial setup, you only need these commands to run the project:

```bash
# Terminal 1 - Backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python scheduler.py

# Terminal 2 - Dashboard  
source venv/bin/activate  # or venv\Scripts\activate on Windows
streamlit run app.py
```

## Troubleshooting

### Common Issues:

**Virtual environment activation fails:**
- Make sure you're in the project root directory
- Check Python installation

**Playwright installation fails:**
- Try running: `pip install --upgrade pip`
- Then retry: `pip install playwright && playwright install`

**Model path error:**
- Ensure the `crypto_model_finetuned` folder exists
- Update the full path in `config/config.yaml`

**Dashboard doesn't load:**
- Check if scheduler is running first
- Ensure database was created with `python create_db.py`

## Project Structure

```
ai-crypto-news-analyzer/
├── app.py                 # Streamlit dashboard
├── scheduler.py           # Main automation script
├── create_db.py          # Database initialization
├── requirements.txt      # Python dependencies
├── .env                  # API keys (create this)
├── config/
│   └── config.yaml      # Configuration file
└── crypto_model_finetuned/  # AI model (add this)
```
