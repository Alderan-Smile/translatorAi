import requests
import json

from model.requestSchema.headerClass import header
from model.requestSchema.payloadChatClass import payloadChat
from model.requestSchema.payloadCompletClass import payloadComplet
from model.responseSchema.responseClass import responseChat, responseComplet, responseError
from model.responseSchema.choicesClass import choice
from model.responseSchema.choicesComplClass import choicesCompl
from model.messageClass import message
from parametros.openRouter import openParam
from model.responseSchema.traduccionClass import traduccion

class postRequest:

    def chatPost(getMessage):
        headers = header()
        payload = payloadChat()
        messagePost = message()
        resp = responseChat()
        errorcont = responseError()
        choicePost = choice()

        headers.Authorization = f"Bearer {openParam.apiKey}"
        headers.Content_Type = openParam.contentType
        payload.model = openParam.modelo
        messagePost.role = openParam.roleUser
        messagePost.content = getMessage
        payload.messages = [messagePost.__dict__]

        response = requests.post(openParam.urlChat, headers=headers.__dict__, data=json.dumps(payload.__dict__))
        jsonResponse = json.loads(response.content.decode('utf-8').strip())

        if response.status_code == 200:
            resp = responseChat.from_dict(jsonResponse)
            print(f"\n\nBot: {resp.choices[0].message.content}\n\n")
        elif response.status_code != 200:
            errorcont = responseError.from_dict(jsonResponse)
            print(f"\n\nError: {errorcont.error.code} \n {errorcont.error.message}\n\n")


    
    def completionPost(getMessage):
        headers = header()
        payload = payloadComplet()
        resp = responseComplet()
        choicePost = choicesCompl()

        headers.Authorization = f"Bearer {openParam.apiKey}"
        headers.Content_Type = openParam.contentType
        payload.model = openParam.modelo
        payload.prompt = getMessage

        response = requests.post(openParam.urlChat, headers=headers.__dict__, data=json.dumps(payload.__dict__))
        jsonResponse = json.loads(response.content.decode('utf-8').strip())

        if response.status_code == 200:
            resp = responseComplet.from_dict(jsonResponse)
            print(f"\n\nBot: {resp.choices[0].text}\n\n")