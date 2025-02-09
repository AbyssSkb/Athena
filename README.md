# 🎤 Athena

> A Chinese voice assistant powered by OpenAI's GPT, featuring wake word detection and web search capabilities.

## ✨ Features

- 🔊 Wake word detection ("Hey Siri")
- 🗣️ Chinese voice command recognition (Google/Azure)
- 🔍 DuckDuckGo web search integration
- 🔊 Multiple TTS engines (pyttsx3, GPT-SoVITS)
- ⏲️ Auto-standby mode

## 🛠️ Prerequisites

- Python 3.10+
- Working microphone and speakers
- [uv package manager ](https://github.com/astral-sh/uv)
- [OpenAI API key](https://openai.com/api/)
- [Picovoice access key](https://console.picovoice.ai/)
- [VLC media player](https://www.videolan.org/)


## 📦 Setup

1. Clone and install:
   ```bash
   git clone https://github.com/AbyssSkb/Athena
   cd Athena
   uv sync
   ```
   
2. Configure environment:
   ```bash
   cp .env.example .env
   ```

   Update `.env` with your credentials:
   ```bash
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   PICOVOICE_ACCESS_KEY="YOUR_PICOVOICE_ACCESS_KEY"
   ```

3. Configure `pyttsx3` for Linux users:
   ```bash
   sudo apt update && sudo apt install espeak-ng libespeak1
   ```

## 🚀 Usage

1. Start: `uv run main.py`
2. Say "Hey Siri" to activate the assistant
3. Speak your command or question in Chinese
4. Say "再见", "退出" or "结束" to end the conversation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
