import os
import time

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
# from elevenlabs.client import ElevenLabs

from fastapi import FastAPI
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    # get_stt_model,
    get_twilio_turn_credentials,
)
from gradio.utils import get_space
from numpy.typing import NDArray
from agent import agent, langfuse_handler
from langchain_core.messages import HumanMessage
from fastrtc.utils import audio_to_bytes, audio_to_float32
from groq import Groq

load_dotenv()

tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
# stt_model = get_stt_model() # default local stt model
# stt_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
groq_stt_client = Groq(api_key=os.getenv("GROQ_API_KEY"))




# See "Talk to Claude" in Cookbook for an example of how to keep
# track of the chat history.
def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []
    # messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    start = time.time()
    # text = stt_model.stt(audio)

    # transcription = stt_client.speech_to_text.convert(
    #     file=audio_to_bytes(audio),
    #     model_id="scribe_v1",
    #     tag_audio_events=False,
    #     language_code="eng",
    #     diarize=False,
    # )
    transcription = groq_stt_client.audio.transcriptions.create(
        file=("audio-file.mp3", audio_to_bytes(audio)),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )
    text = transcription.text
    print("transcription", time.time() - start)
    print("prompt", text)
    chatbot.append({"role": "user", "content": text})
    yield AdditionalOutputs(chatbot)
    # messages.append({"role": "user", "content": text})
    # response_text = (
    #     groq_client.chat.completions.create(
    #         model="llama-3.1-8b-instant",
    #         max_tokens=200,
    #         messages=messages,  # type: ignore
    #     )
    #     .choices[0]
    #     .message.content
    # )
    inputs = {"messages": [HumanMessage(text)]}
    result = agent.invoke(inputs, config={"configurable": {"thread_id": "conversation1"}})
    response_text = result['messages'][-1].content

    chatbot.append({"role": "assistant", "content": response_text})

    for i, chunk in enumerate(
        tts_client.text_to_speech.convert_as_stream(
            text=response_text,  # type: ignore
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",
        )
    ):
        if i == 0:
            yield AdditionalOutputs(chatbot)
        audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
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
    ui_args={"title": "LLM Voice Chat (ElevenLabs, and WebRTC ⚡️)"},
)

# Mount the STREAM UI to the FastAPI app
# Because I don't want to build the UI manually
app = FastAPI()
app = gr.mount_gradio_app(app, stream.ui, path="/")


if __name__ == "__main__":
    import os

    os.environ["GRADIO_SSR_MODE"] = "false"

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        stream.ui.launch(server_port=7860)
