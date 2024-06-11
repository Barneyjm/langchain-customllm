from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def read_item(prompt: str):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": prompt,
                    "role": "assistant"
                }
            }
        ],
        "created": 1677664795,
        "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
        "model": "gpt-3.5-turbo-0613",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 17,
            "prompt_tokens": 57,
            "total_tokens": 74
        }
    }