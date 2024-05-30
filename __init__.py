from quart import Quart, request, render_template, g, send_file
from functools import wraps
from deepl_tools import DeepLTranslator
from chatgpt_tools import ChatgptTranslator

app = Quart(__name__)

# 初始化 Translator
def init_translator_chatgpt(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not hasattr(g, 'chatgpt'):
            print("ChatGPT初始化中...")
            g.chatgpt = ChatgptTranslator()
            print("翻譯器成功初始化。")
        return await func(*args, **kwargs) 
    return wrapper

def init_translator_deepl(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not hasattr(g, 'deepl'):
            print("DeepL初始化中...")
            g.deepl = DeepLTranslator()
            print("翻譯器成功初始化。")
        return await func(*args, **kwargs) 
    return wrapper