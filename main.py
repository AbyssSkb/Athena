import pyttsx3
import speech_recognition as sr
import numpy as np
import pvporcupine
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI
import openai
from io import BytesIO
import tempfile
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from datetime import datetime
import asyncio
import httpx
import urllib.parse
import json
import time
import vlc
import os

# 加载环境变量
load_dotenv()
# OpenAI API密钥
openai_api_key = os.getenv("OPENAI_API_KEY")
# OpenAI API基础URL，默认为官方API地址
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# 使用的OpenAI聊天模型，默认为gpt-4o-mini
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
openai_transcribe_model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
openai_speech_model = "RVC-Boss/GPT-SoVITS"
openai_speech_voice = "RVC-Boss/GPT-SoVITS:diana"
# Picovoice唤醒词检测服务的访问密钥
picovoice_access_key = os.getenv("PICOVOICE_ACCESS_KEY")
# 语音识别引擎选择，支持'google'或'azure'，默认为google
recognizer_engine = os.getenv("RECOGNIZER_ENGINE", "google")
# 语音合成引擎选择，支持'pyttsx3'或'gsv'，默认为pyttsx3
speaker_engine = os.getenv("SPEAKER_ENGINE", "pyttsx3")
# GPT-SoVITS服务的基础URL，用于gsv语音合成
gsv_base_url = os.getenv("GSV_BASE_URL")
# GPT-SoVITS的参考音频路径
ref_audio_path = os.getenv("REF_AUDIO_PATH")
# GPT-SoVITS的提示文本
prompt_text = os.getenv("PROMPT_TEXT")
# GPT-SoVITS的提示语言，默认为auto
prompt_lang = os.getenv("PROMPT_LANG", "auto")
# GPT-SoVITS的文本语言，默认为auto
text_lang = os.getenv("TEXT_LANG", "auto")

# 初始化
prompt_text = urllib.parse.quote(prompt_text)
ref_audio_path = urllib.parse.quote(ref_audio_path, safe="")
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty("voices")
tts_engine.setProperty("voice", voices[-1].id)
porcupine = pvporcupine.create(keywords=["hey siri"], access_key=picovoice_access_key)
recognizer = sr.Recognizer()
recognizer.operation_timeout = 5
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 1.5
chat_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
transcribe_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
speech_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
conversation_history = []
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "使用DuckDuckGo搜索引擎查询信息。可以搜索最新新闻、文章、博客等内容。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "搜索的关键词列表。例如：['Python', '机器学习', '最新进展']。",
                    }
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "获取当前的日期和时间",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "addtionalProperties": False,
            },
        },
    },
]


async def fetch_url(url: str) -> str:
    """异步获取指定URL的网页内容并提取文本。

    Args:
        url (str): 要获取内容的网页URL

    Returns:
        str: 提取的网页文本内容，经过清理和格式化（移除链接标签、图片标签，保留纯文本）

    Example:
        >>> async def main():
        ...     text = await fetch_url('https://example.com')
        ...     print(text)
        >>> asyncio.run(main())
        'Example Domain...'
    """
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0"
            }
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a"):
                a.unwrap()
            for img in soup.find_all("img"):
                img.decompose()
            text = soup.get_text()
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            return text
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
            return ""
        except httpx.HTTPStatusError as exc:
            print(
                f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
            )
            return ""


async def crawl_web(urls: list[str]) -> list[str]:
    """并发爬取多个URL的内容。

    Args:
        urls (list[str]): 要爬取的URL列表

    Returns:
        list[str]: 所有URL的爬取结果列表，每个元素为对应URL的文本内容
    """
    results = await asyncio.gather(*(fetch_url(url) for url in urls))
    length = len("".join(results))
    print(f"Context Length: {length}")
    return results


def search_duckduckgo(keywords: list[str]) -> list[str]:
    """使用 DuckDuckGo 搜索引擎执行查询并获取详细内容。

    Args:
        keywords (list[str]): 搜索关键词列表，每个元素为一个关键词

    Returns:
        list[str]: 搜索结果页面的完整文本内容列表。每个元素对应一个搜索结果页面的文本

    Example:
        >>> results = search_duckduckgo(['Python', '机器学习'])
        >>> print(f"找到{len(results)}个结果")
        找到3个结果
    """
    search_term = " ".join(keywords)
    print(f"Searching: {search_term}")
    results = DDGS().text(
        keywords=search_term,
        region="cn-zh",
        safesearch="on",
        max_results=3,
        backend="html",
    )
    for i, result in enumerate(results, start=1):
        print(f"Index {i}: {result['href']} {result['title']}")

    urls = [result["href"] for result in results]
    response_results = asyncio.run(crawl_web(urls))
    return response_results


def get_current_datetime() -> str:
    """获取当前的日期和时间。

    Returns:
        str: 当前日期和时间的字符串表示
              格式示例: "2024-02-20 15:30:45.123456"

    Example:
        >>> get_current_datetime()
        '2024-02-20 15:30:45.123456'
    """
    print("获取当前的日期和时间")
    now = datetime.now()
    return str(now)


def call_function(name: str, args: dict):
    """根据函数名和参数动态调用工具函数。

    Args:
        name (str): 要调用的函数名，必须是已注册的工具函数之一
        args (dict): 函数参数字典，必须与目标函数参数匹配

    Returns:
        Any: 函数调用的结果，类型取决于被调用的具体函数
            - search_duckduckgo: 返回 list[str]
            - get_current_datetime: 返回 str

    Raises:
        ValueError: 当函数名未找到时抛出

    Example:
        >>> time_str = call_function('get_current_datetime', {})
        >>> print(time_str)  # 2024-02-20 15:30:45.123456
        >>> results = call_function('search_duckduckgo', {'keywords': ['Python']})
        >>> print(len(results))  # 3
    """
    match name:
        case "search_duckduckgo":
            return search_duckduckgo(**args)
        case "get_current_datetime":
            return get_current_datetime(**args)
        case other:
            raise ValueError(f"Unknown function name: {other}")


def detect_wake_word() -> None:
    """监听并检测唤醒词。

    使用 Picovoice Porcupine 引擎监听特定的唤醒词（"hey siri"）。
    当检测到唤醒词时，会播放语音反馈并退出监听循环。

    Note:
        需要预先配置 PICOVOICE_ACCESS_KEY 环境变量
    """
    try:
        pyaudio_engine = pyaudio.PyAudio()
        stream = pyaudio_engine.open(
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
    finally:
        stream.stop_stream()
        stream.close()
        pyaudio_engine.terminate()


def listen_for_commands() -> str:
    """监听并转换用户的语音指令为文本。

    使用选定的语音识别引擎（通过RECOGNIZER_ENGINE环境变量配置）：
    - google: Google Speech Recognition，无需配置额外凭据
    - azure: Microsoft Azure Speech Recognition，需要配置以下环境变量：
        - AZURE_KEY: Azure Speech Service 的访问密钥
        - AZURE_LOCATION: 服务区域（默认为 "eastus"）

    Returns:
        str: 识别出的用户语音指令文本

    Raises:
        sr.UnknownValueError: 当语音无法被识别时抛出
        sr.RequestError: 当语音识别服务出现问题时抛出
        sr.WaitTimeoutError: 当等待用户输入超过20秒时抛出

    Example:
        >>> try:
        ...     command = listen_for_commands()
        ...     print(f"识别到的命令: {command}")
        ... except sr.UnknownValueError:
        ...     print("无法识别语音")
    """
    print("等待用户指令")
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=20)
    print("收到指令")

    match recognizer_engine:
        case "google":
            command = recognizer.recognize_google(audio, language="zh-CN")
        case "azure":
            azure_key = os.getenv("AZURE_KEY")
            azure_location = os.getenv("AZURE_LOCATION", "eastus")
            command = recognizer.recognize_azure(
                audio,
                key=azure_key,
                location=azure_location,
                language="zh-CN",
            )[0]
        case "openai":
            audio_file = BytesIO(audio.get_wav_data())
            audio_file.name = "SpeechRecognition_audio.wav"
            transcription = transcribe_client.audio.transcriptions.create(
                model=openai_transcribe_model, file=audio_file
            )
            command = transcription.text

    print(f"User: {command}")
    return command


def speak(text: str):
    """使用文字转语音引擎朗读文本。

    支持两种语音合成引擎（通过SPEAKER_ENGINE环境变量配置）：
    - pyttsx3: 使用本地TTS引擎，无需网络连接
    - gsv: 使用远程GPT-SoVITS服务，需要配置以下环境变量：
        - GSV_BASE_URL: GPT-SoVITS服务的基础URL
        - REF_AUDIO_PATH: 参考音频路径
        - PROMPT_TEXT: 提示文本
        - PROMPT_LANG: 提示语言（默认为"auto"）
        - TEXT_LANG: 文本语言（默认为"auto"）

    Args:
        text (str): 需要朗读的文本内容

    Raises:
        RuntimeError: 当语音合成失败时抛出
        ConnectionError: 当使用gsv引擎且网络连接失败时抛出

    Example:
        >>> speak("现在是下午三点钟")
        Assistant: 现在是下午三点钟
        # 语音输出："现在是下午三点钟"
    """
    print(f"Assistant: {text}")
    match speaker_engine:
        case "pyttsx3":
            tts_engine.say(text)
            tts_engine.runAndWait()
        case "gsv":
            text = urllib.parse.quote(text)
            audio_url = f"{gsv_base_url}?text={text}&text_lang={text_lang}&ref_audio_path={ref_audio_path}&prompt_text={prompt_text}&prompt_lang={prompt_lang}&streaming_mode=true&text_split_method=cut0"
            player = vlc.MediaPlayer(audio_url)
            player.play()

            while player.get_state() not in [vlc.State.Ended, vlc.State.Error]:
                time.sleep(0.5)
        case "openai":
            temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_audio_path = temp_audio_file.name
            with speech_client.audio.speech.with_streaming_response.create(
                model=openai_speech_model,
                voice=openai_speech_voice,
                input=text,
                response_format="mp3",
            ) as response:
                response.stream_to_file(temp_audio_path)

            player = vlc.MediaPlayer(temp_audio_path)
            player.play()

            while player.get_state() not in [vlc.State.Ended, vlc.State.Error]:
                time.sleep(0.5)


def single_chat_completion():
    """执行单次与OpenAI API的对话请求。

    使用当前的对话历史和工具配置，向OpenAI API发送请求并获取回复。

    Returns:
        ChatCompletion: OpenAI API的响应对象，包含模型的回复和可能的工具调用

    Raises:
        TimeoutError: 当请求超时（5秒）时抛出

    Note:
        需要预先配置以下环境变量：
        - OPENAI_API_KEY: OpenAI API密钥
        - OPENAI_BASE_URL: OpenAI API 基础URL（默认为"https://api.openai.com/v1"）
        - OPENAI_MODEL: 使用的模型名称（默认为"gpt-4-mini"）
    """
    print("正在询问AI")
    try:
        completion = chat_client.chat.completions.create(
            model=openai_chat_model,
            messages=conversation_history,
            tools=tools,
            # timeout=10,
        )
        conversation_history.append(completion.choices[0].message)
        print("询问完毕")
        return completion
    except openai.APITimeoutError as e:
        print("API请求错误")
        raise e


def get_model_response(user_input: str) -> str:
    """调用 OpenAI API 获取对话回复。

    处理用户输入，管理对话历史，处理可能的工具调用，并返回最终的AI回复。

    Args:
        user_input (str): 用户的输入文本

    Returns:
        str: AI模型生成的回复文本

    Example:
        >>> response = get_model_response("现在几点了？")
        >>> print(response)
        现在是下午三点二十分
    """
    conversation_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    completion = single_chat_completion()
    while completion.choices[0].message.tool_calls:
        for tool_call in completion.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            result = call_function(name, args)
            result = json.dumps(result, ensure_ascii=False)
            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
        completion = single_chat_completion()

    model_response = completion.choices[0].message.content.strip()
    return model_response


def start_assistant() -> None:
    """启动语音助手的主循环。

    实现以下功能：
    1. 等待唤醒词激活
    2. 监听用户语音输入
    3. 处理语音指令并生成回复
    4. 通过语音输出回复
    5. 处理错误和异常情况

    支持的退出命令：
    - "再见"
    - "退出"
    - "结束"

    Note:
        使用Ctrl+C可以强制退出程序
        连续3次识别失败将自动进入待机状态
    """
    speak("语音助手已开机")
    print(f"Chat model: {openai_chat_model}")
    print(f"Recognizer engine: {recognizer_engine}")
    print(f"Speaker engine: {speaker_engine}")
    try:
        while True:
            conversation_history.clear()
            print("进入待机状态，等待唤醒")
            detect_wake_word()
            unsuccessful_tries = 0
            conversation_history.append(
                {
                    "role": "system",
                    "content": f"""
                    你是一个友好的AI语音助手，和用户进行日常对话聊天，请用简洁的语言回答用户的问题。
                    当前的日期和时间为{get_current_datetime()}
                    """,
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
                    unsuccessful_tries = 0
                except sr.UnknownValueError:
                    speak("抱歉，我没有听清楚，请再说一遍。")
                    unsuccessful_tries += 1
                except sr.WaitTimeoutError:
                    speak("进入待机状态。")
                    break
                except Exception as e:
                    print(f"{type(e).__name__}: {e}")
                    unsuccessful_tries += 1
                    speak("发生错误")
                if unsuccessful_tries >= 3:
                    speak("尝试次数过多，进入待机状态。")
                    break

    except KeyboardInterrupt:
        speak("退出程序")


start_assistant()
