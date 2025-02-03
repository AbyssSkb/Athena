# 🎤 Athena

> A Chinese voice assistant powered by OpenAI's GPT and Python text-to-speech technology, featuring wake word detection.

## ✨ Features

- 🔊 Wake word detection ("Hey Siri")
- 🗣️ Chinese voice command recognition
- 🤖 AI-powered conversations using OpenAI GPT
- 🎯 Automatic ambient noise adjustment
- 🔄 Conversation history management
- ⏲️ Auto-standby mode
- 🎵 Text-to-speech response in Chinese

## 🛠️ Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager 
- OpenAI API key
- [Picovoice Access key](https://console.picovoice.ai/)
- Working microphone and speakers

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/AbyssSkb/Athena
cd Athena
```

2. Install required packages:
```bash
uv sync
```

## ⚙️ Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Update `.env` with your credentials:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: OpenAI API base URL
- `OPENAI_MODEL`: GPT model to use
- `PICOVOICE_ACCESS_KEY`: Your Picovoice access key

3. Configure `pyttsx3` for Linux user:
```bash
sudo apt update && sudo apt install espeak-ng libespeak1
```
> **Note:** This step is only required for Linux user if voice output is not working.

## 🚀 Usage

1. Start the voice assistant:
```bash
uv run main.py
```

2. Say "Hey Siri" to activate the assistant
3. Speak your command or question
4. Say "再见", "退出" or "结束" to end the conversation

## ⚡ Quick Commands

- Wake Word: "Hey Siri"
- Exit Commands: "再见", "退出", "结束"

## 🛡️ Error Handling

- Automatic retry on speech recognition failures (max 3 attempts)
- Ambient noise adjustment
- Timeout handling with auto-standby mode

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
