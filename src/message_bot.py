import json
import requests


class Bot:
    def __init__(self, url):
        self.url = url
        self.headers = {'Content-Type': 'application/json'}
        
    def text(self, content):
        data = {
                "msg_type": "text",
                "content": {
                    "text": content
                }
            }
        requests.post(self.url, headers=self.headers, data=json.dumps(data))
            
    
    def warn(self, content):
        data = {
            "msg_type": "text",
            "content": {
                "text": content + '\n<at user_id="all">所有人</at>'
                }
            }
        requests.post(self.url, headers=self.headers, data=json.dumps(data))


# url = "https://open.feishu.cn/open-apis/bot/v2/hook/c5634e62-18fa-45e7-9af3-a2dfea7be4eb"
# url_private = "https://open.feishu.cn/open-apis/bot/v2/hook/4519b97c-d166-430f-87bc-13a6b8d35dac"

# # 例子：
# my_url = url_private
# my_bot = Bot(my_url)
# a = 12345
# my_msg = f"Hello, test{a}"
# my_bot.text(my_msg)

