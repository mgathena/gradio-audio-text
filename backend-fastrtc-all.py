import os

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import ReplyOnPause, Stream, get_twilio_turn_credentials
from fastrtc.utils import audio_to_bytes
from gradio.utils import get_space
from groq import Groq
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from numpy.typing import NDArray
from pydantic import BaseModel

from agent import agent

load_dotenv()

tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
groq_stt_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Global variable to store the agent instance


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Initialize the agent on startup
#     global _agent
#     _agent = agent
#     yield
#     # Clean up resources on shutdown
#     _agent = None


class Query(BaseModel):
    question: str


full_response = ""


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []

    transcription = groq_stt_client.audio.transcriptions.create(
        file=("audio-file.mp3", audio_to_bytes(audio)),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )

    question = transcription.text
    #question = "Which sales agent made the most in sales in 2009?"  # debug
    chatbot.append({"role": "user", "content": question})
    chatbot.append({"role": "assistant", "content": ""})

    def text_stream():
        global full_response
        full_response = ""
        inputs = {"messages": [HumanMessage(question)]}
        config = {
            "configurable": {"thread_id": 1},
        }
        prev_type = None
        for chunk in agent.stream(inputs, config=config, stream_mode="messages"):
            message, metadata = chunk
            if message.response_metadata.get("finish_reason") == "stop":
                break
            if isinstance(message, AIMessageChunk):
                if message.additional_kwargs:  # this means a tool call is in progress
                    current_type = "tool_call"
                    if prev_type != current_type:
                        yield f"\nFunction Called: {message.additional_kwargs['tool_calls'][0]['function']['name']}\n"
                        prev_type = current_type
                    # print(message.content, end="|", flush=True)
                    yield message.additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                else:  # this means an ai response is in progress
                    # breakpoint()
                    # if message.response_metadata["finish_reason"] != "tool_calls":
                    if not message.response_metadata:
                        current_type = "ai_message"
                        if prev_type != current_type:
                            prev_type = current_type
                            yield "\nAI Response:\n"
                        full_response += message.content
                        chatbot[-1]["content"] = full_response
                        print(full_response)
                        yield message.content
            if isinstance(message, ToolMessage):
                current_type = "tool"
                if prev_type != current_type:
                    prev_type = current_type
                    yield "\nTool Response:\n"
                yield message.content

    chatbot.append({"role": "assistant", "content": full_response + " "})
    audio_stream = tts_client.generate(
        text=text_stream(),
        # voice="Rachel",  # Cassidy is also really good
        voice="JBFqnCBsd6RMkjVDRZzb",
        voice_settings=VoiceSettings(
            similarity_boost=0.9, stability=0.6, style=0.4, speed=1
        ),
        model="eleven_multilingual_v2",
        output_format="pcm_24000",
        stream=True,
    )

    for audio_chunk in audio_stream:
        audio_array = (
            np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )
        yield (24000, audio_array)


chatbot = gr.Chatbot(type="messages")


stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response, input_sample_rate=16000),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
    ui_args={"title": "NL2SQL Voice Chat (LangGraph, ElevenLabs, and Groq ⚡️)"},
)


app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app = gr.mount_gradio_app(app, stream.ui, path="/")


if __name__ == "__main__":
    # import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    import os

    os.environ["GRADIO_SSR_MODE"] = "false"
    stream.ui.launch(server_port=7861)
