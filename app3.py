import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, request, abort, Response
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, ReplyMessageRequest,
    TextMessage, QuickReply, QuickReplyItem, MessageAction
)
import ollama
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 載入環境變量
load_dotenv()

# 從JSON文件載入配置
try:
    with open('config.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
except json.JSONDecodeError as e:
    logger.error(f"JSON 解析錯誤: {str(e)}")
    raise
except UnicodeDecodeError as e:
    logger.error(f"編碼錯誤: {str(e)}。請確保 config.json 文件使用 UTF-8 編碼保存。")
    raise
except FileNotFoundError:
    logger.error("找不到 config.json 文件。請確保該文件在正確的路徑中。")
    raise

# 全局配置
SYSTEM_PROMPT = config['SYSTEM_PROMPT']
OLLAMA_MODEL = config['OLLAMA_MODEL']
QUICK_REPLY_OPTIONS = config['QUICK_REPLY_OPTIONS']
PREDEFINED_TOPICS = config['PREDEFINED_TOPICS']
ENABLE_QUICK_REPLY = True


product_file = config.get("PRODUCT_FILE")
if product_file:
    try:
        with open(product_file, 'r', encoding='utf-8') as file:
            product_list = file.read()
            config["PRODUCT_LIST"] = product_list  # 將商品目錄內容加入設定中
    except FileNotFoundError:
        print(f"檔案 '{product_file}' 找不到。請確認檔案名稱與路徑是否正確。")


logger.info(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
logger.info(f"ENABLE_QUICK_REPLY: {ENABLE_QUICK_REPLY}")


# 初始化 Flask 應用
app = Flask(__name__)

# 初始化 Line Bot API
configuration = Configuration(access_token='rjf/BEsBX6BHmBH85EDls5M42xE5OQ9Hb19pn30P1F51HZxDxzUdFNirvG9HAOXU2VqPsD9mmkRNt9C4nOeMjI5s/Kk5+ZywfdkKB5I6iT5fc5wI4u7G8cvKJ9s6myXO9IP4pqDed9iIWOBrjmcG9QdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('5426a38c70c53f34ccdfd2fac35dba38')


# 設置 Prometheus 指標
MESSAGES_PROCESSED = Counter('messages_processed_total', 'Total number of messages processed')
IMAGE_GENERATIONS = Counter('image_generations_total', 'Total number of image generations')
RESPONSE_TIME = Histogram('response_time_seconds', 'Response time in seconds')

class Memory:
    def __init__(self, system_message, memory_message_count=config['MEMORY_MESSAGE_COUNT']):
        self.system_message = system_message
        self.memory_message_count = memory_message_count
        self.memories = {}
        self.pending_responses = {}  # 儲存待發送的分段回覆

    def append(self, user_id, role, content):
        if user_id not in self.memories:
            self.memories[user_id] = []
        self.memories[user_id].append({"role": role, "content": content})
        self.memories[user_id] = self.memories[user_id][-self.memory_message_count:]

    def get(self, user_id):
        return [{"role": "system", "content": self.system_message}] + self.memories.get(user_id, [])

    def change_system_message(self, new_system_message):
        self.system_message = new_system_message

    def set_pending_response(self, user_id, response_segments):
        self.pending_responses[user_id] = response_segments

    def get_next_segment(self, user_id):
        if user_id in self.pending_responses and self.pending_responses[user_id]:
            return self.pending_responses[user_id].pop(0)
        return None

# 初始化記憶系統
memory = Memory(system_message=SYSTEM_PROMPT)

def create_text_message(text, quick_reply=None):
    if ENABLE_QUICK_REPLY and quick_reply:
        return TextMessage(text=text, quick_reply=quick_reply)
    return TextMessage(text=text)

def create_quick_reply():
    if not ENABLE_QUICK_REPLY:
        return None
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label=label, text=label))
        for label in QUICK_REPLY_OPTIONS
    ])

def create_continue_button():
    return TextMessage(text="請點擊 '繼續' 查看下一部分。", quick_reply=QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="繼續", text="繼續"))
    ]))

def format_assistant_response(response):
    return f"以下是我的回應：\n\n{response}\n\n還有什麼我可以幫助您的嗎？"

def segment_response(response, max_length=200):
    words = response.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length and current_segment:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0
        current_segment.append(word)
        current_length += len(word) + 1

    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

def ollama_chat_completions(messages):
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama API 請求失敗: {str(e)}")
        return None


def handle_admin_commands(text):
    global OLLAMA_MODEL, ENABLE_IMAGE_GENERATION
    if text.startswith('/更新系統提示 '):
        new_system_prompt = text[8:].strip()
        memory.change_system_message(new_system_prompt)
        return [create_text_message("系統提示已更新。", create_quick_reply())]
    elif text.startswith('/更改模型 '):
        new_model = text[6:].strip()
        OLLAMA_MODEL = new_model
        logger.info(f"Ollama 模型已更新為: {OLLAMA_MODEL}")
        return [create_text_message(f'Ollama 模型已更新為 {new_model}。', create_quick_reply())]
    elif text == '/當前模型':
        return [create_text_message(f'當前使用的 Ollama 模型是 {OLLAMA_MODEL}。', create_quick_reply())]
   

def handle_predefined_topics(text, user_id):
    memory.append(user_id, 'user', PREDEFINED_TOPICS[text])
    chat_history = memory.get(user_id)
    response = ollama_chat_completions(chat_history)
    
    if not response:
        raise Exception("無法獲得回應")
    
    response_segments = segment_response(response)
    memory.set_pending_response(user_id, response_segments)
    
    first_segment = memory.get_next_segment(user_id)
    messages = [create_text_message(first_segment)]
    
    if len(response_segments) > 1:
        messages.append(create_continue_button())
    elif ENABLE_QUICK_REPLY:
        messages[0] = create_text_message(first_segment, create_quick_reply())
    
    memory.append(user_id, 'assistant', response)
    return messages

def handle_general_conversation(text, user_id):
    memory.append(user_id, 'user', text)
    chat_history = memory.get(user_id)
    response = ollama_chat_completions(chat_history)
    
    if not response:
        raise Exception("無法獲得回應")
    
    response_segments = segment_response(response)
    memory.set_pending_response(user_id, response_segments)
    
    first_segment = memory.get_next_segment(user_id)
    messages = [create_text_message(first_segment)]
    
    if len(response_segments) > 1:
        messages.append(create_continue_button())
    elif ENABLE_QUICK_REPLY:
        messages[0] = create_text_message(first_segment, create_quick_reply())
    
    memory.append(user_id, 'assistant', response)
    return messages

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("無效的簽名。請檢查您的頻道訪問令牌/頻道密鑰。")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    logger.info(f'{user_id}: {text}')

    MESSAGES_PROCESSED.inc()

    try:
        with RESPONSE_TIME.time():
            if text == "繼續":
                next_segment = memory.get_next_segment(user_id)
                if next_segment:
                    messages = [create_text_message(next_segment)]
                    if memory.pending_responses.get(user_id):
                        messages.append(create_continue_button())
                    elif ENABLE_QUICK_REPLY:
                        messages[0] = create_text_message(next_segment, create_quick_reply())
                else:
                    messages = [create_text_message("回覆已結束。還有什麼我可以幫您的嗎？", create_quick_reply() if ENABLE_QUICK_REPLY else None)]
            elif text == '/選項':
                messages = [create_text_message("以下是一些選項。我今天能為您做些什麼？", create_quick_reply())]
            elif text.startswith('/'):
                messages = handle_admin_commands(text)
            elif text in PREDEFINED_TOPICS:
                messages = handle_predefined_topics(text, user_id)
            else:
                messages = handle_general_conversation(text, user_id)

            # 發送消息
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=messages))

    except Exception as e:
            logger.error(f"處理消息時出錯: {str(e)}")
            error_message = create_text_message(f"抱歉，似乎出現了一個問題：{str(e)}。讓我們稍後再試。您現在需要其他協助嗎？", create_quick_reply() if ENABLE_QUICK_REPLY else None)
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[error_message]))

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    port = int(config.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)