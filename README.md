# 🎤 Athena

> A Chinese voice assistant powered by OpenAI's GPT, featuring wake word detection and web search capabilities.

## ✨ Features

- 🔊 Wake word detection ("Hey Siri")
- 🗣️ Chinese voice command recognition (Google/Azure)
- 🔍 DuckDuckGo web search integration
- 🔊 Multiple TTS engines (pyttsx3, GPT-SoVITS)
- 🔄 Conversation history management
- ⏲️ Auto-standby mode

## 🛠️ Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager 
- OpenAI API key
- [Picovoice Access key](https://console.picovoice.ai/)
- Azure Speech API key (optional, if using Azure speech recognition)
- GPT-SoVITS server (optional, if using GPT-SoVITS TTS)
- VLC media player
- Working microphone and speakers

## 📦 Setup

1. Clone and install:
```bash
git clone https://github.com/AbyssSkb/Athena
cd Athena
uv sync
```

2. Install VLC media player:
   - Windows: Download and install from [VideoLAN official website](https://www.videolan.org/)
   - Linux: `sudo apt install vlc`
   - macOS: `brew install vlc`

3. Configure environment:
```bash
cp .env.example .env
```

Update `.env` with your credentials:
```bash
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
PICOVOICE_ACCESS_KEY="YOUR_PICOVOICE_ACCESS_KEY"
```

4. Configure `pyttsx3` for Linux users:
```bash
sudo apt update && sudo apt install espeak-ng libespeak1
```
> **Note:** This step is only required for Linux users if voice output is not working.

## 🚀 Usage

1. Start: `uv run main.py`
2. Say "Hey Siri" to activate the assistant
3. Speak your command or question in Chinese
4. Say "再见", "退出" or "结束" to end the conversation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
