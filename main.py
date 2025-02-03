import pyttsx3
import speech_recognition as sr
import numpy as np
import pvporcupine
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI
import os

# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
chat_model = os.getenv("OPENAI_MODEL")
access_key = os.getenv("PICOVOICE_ACCESS_KEY")

# 初始化
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[-1].id)
porcupine = pvporcupine.create(keywords=["hey siri"], access_key=access_key)
recognizer = sr.Recognizer()
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
conversation_history = []


def detect_wake_word():
    """
    监听并检测唤醒词，当检测到指定唤醒词时函数返回。

    使用 Porcupine 热词检测引擎来识别唤醒词 "hey siri"。
    检测到唤醒词后会播放语音确认，并关闭音频流。

    Returns:
        None

    Raises:
        PyAudioError: 当音频设备初始化失败时抛出
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )
    while True:
        pcm = stream.read(porcupine.frame_length)
        pcm = np.frombuffer(pcm, dtype=np.int16)
        result = porcupine.process(pcm)
        if result >= 0:
            speak("我在")
            break
    stream.stop_stream()
    stream.close()
    p.terminate()


def listen_for_commands() -> str:
    """
    监听并转换用户的语音指令为文本。

    使用 Google Speech Recognition 服务将音频转换为文本。
    会自动调整环境噪声水平，超时时间为20秒。

    Returns:
        str: 转换后的用户语音指令文本

    Raises:
        UnknownValueError: 当语音无法被识别时抛出
        RequestError: 当语音识别服务出现问题时抛出
        WaitTimeoutError: 当等待超时时抛出
    """
    print("等待用户指令")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=20)

    command = recognizer.recognize_google(audio, language="zh-CN")
    print(f"你: {command}")
    return command


def speak(sentence: str):
    """
    使用文字转语音引擎朗读指定文本。

    Parameters:
        sentence (str): 需要朗读的文本内容

    Returns:
        None

    Example:
        >>> speak("你好")
        助手：你好
    """
    print(f"助手：{sentence}")
    engine.say(sentence)
    engine.runAndWait()


def get_model_response(user_input: str) -> str:
    """
    调用 OpenAI API 获取对话回复。

    将用户输入添加到对话历史中，通过 GPT 模型生成回复，
    并将回复也保存到对话历史中。

    Parameters:
        user_input (str): 用户的输入文本

    Returns:
        str: AI 模型生成的回复文本

    Raises:
        Exception: 当 API 调用失败时返回错误信息
    """
    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=conversation_history,
        )
        ai_response = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        print(f"Error getting model response: {e}")
        return "抱歉，我现在无法回答这个问题。"


def start_assistant():
    """
    启动语音助手的主循环。

    实现以下功能：
    1. 等待唤醒词激活
    2. 开始对话循环，接收用户指令
    3. 处理用户指令并返回响应
    4. 处理各种异常情况

    Features:
        - 自动清理对话历史
        - 支持退出命令词（"再见"、"退出"、"结束"）
        - 错误重试机制（最多3次）
        - 超时自动进入待机

    Returns:
        None

    Raises:
        KeyboardInterrupt: 当用户手动中断程序时
    """
    speak("语音助手已开机")
    try:
        while True:
            conversation_history.clear()
            detect_wake_word()
            unsuccessful_tries = 0
            conversation_history.append(
                {
                    "role": "developer",
                    "content": "你是一个友好的AI助手，请用简洁的语言回答用户的问题。",
                }
            )

            while True:
                try:
                    user_input = listen_for_commands()
                    if any(word in user_input for word in ["再见", "退出", "结束"]):
                        speak("好的，下次再见。")
                        break

                    response = get_model_response(user_input)
                    speak(response)
                except sr.UnknownValueError:
                    speak("抱歉，我没有听清楚，请再说一遍。")
                    unsuccessful_tries += 1
                except sr.RequestError as e:
                    speak(f"语音识别服务错误: {e}")
                    unsuccessful_tries += 1
                except sr.WaitTimeoutError:
                    speak("进入待机状态。")
                    break

                if unsuccessful_tries >= 3:
                    speak("尝试次数过多，进入待机状态。")
                    break

    except KeyboardInterrupt:
        speak("退出程序")


start_assistant()
