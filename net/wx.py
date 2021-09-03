import json
import urllib.request
import urllib.parse


def send_wx(content):
    data = {
        'msgtype': 'text',
        'text': content,
    }

    json_data = json.dumps(data).encode('utf8')

    req = urllib.request.Request(
        'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=0d1010c7-acd1-4c3a-b160-7483d9f00f5c',
        data=json_data,
        headers={'content-type': 'application/json'})

    response = urllib.request.urlopen(req)
    return response.read().decode('utf8')
