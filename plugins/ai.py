"""AI — Chat, Images, and Transcription helpers

Usage:
    {i}ai <prompt>
        Send a chat prompt to the configured OpenAI-compatible endpoint and reply with the result.

    {i}ai create <prompt>
        Generate an image from the prompt and send it to the chat.

    Reply to media with {i}ai <instruction>:
        • Reply to an image/photo/sticker: edits the image using the prompt.
        • Reply to a voice/audio message: transcribes (via Whisper) and optionally answers questions about the audio.
        • Reply to a text message: analyzes the text or suggests a short reply.

Configuration (set with `setdb`):
    • AI_API_KEY — API key for the OpenAI-compatible provider (preferred; can be changed at runtime).
    • OPENAI_URL — Base URL for OpenAI-compatible endpoints (e.g. https://api.openai.com).
    • OPENAI_MODEL — Optional model id.

Examples:
    {i}setdb AI_API_KEY sk-xxx
    {i}ai write a short poem about coffee
    {i}ai create a fantasy landscape with a castle

This docstring is used by the help system; {i} will be replaced with the command prefix.
"""

import os
import json
import aiohttp
import base64
import tempfile

from . import async_searcher, udB, ultroid_cmd, eor, LOGS


def get_api_key():
    # Prefer API key stored in bot DB so it can be updated at runtime via `setdb AI_API_KEY <key>`.
    db_key = udB.get_key("AI_API_KEY") or udB.get_key("OPENAI_API_KEY")
    if db_key:
        return db_key
    return os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY")


def get_api_base():
    base = udB.get_key("OPENAI_URL")
    if not base:
        base = "https://api.openai.com"
    return base.rstrip("/")


def get_model():
    return udB.get_key("OPENAI_MODEL") or "provider-3/gpt-4.1-nano"


async def _chat_completion(prompt: str):
    api_key = get_api_key()
    if not api_key:
        return "⚠️ AI API key not set. Set it using `setdb AI_API_KEY your_api_key` or export `AI_API_KEY` in your .env."
    base = get_api_base()
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": get_model(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1500,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload, headers=headers, timeout=120) as resp:
                data = await resp.json()
    except Exception as e:
        LOGS.exception(e)
        return f"Error contacting AI API: {e}"

    # Try standard OpenAI response shape
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(data)


async def _image_create(prompt: str):
    api_key = get_api_key()
    if not api_key:
        return None, "⚠️ AI API key not set. Set it using `setdb AI_API_KEY your_api_key` or export `AI_API_KEY` in your .env."
    base = get_api_base()
    url = f"{base}/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "n": 1, "size": "1024x1024"}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload, headers=headers, timeout=120) as resp:
                data = await resp.json()
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error contacting image API: {e}"

    # data may contain data[0].b64_json or data[0].url
    try:
        if data and isinstance(data, dict) and "data" in data and data["data"]:
            item = data["data"][0]
            if item.get("b64_json"):
                b = base64.b64decode(item["b64_json"])
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tf.write(b)
                tf.close()
                return tf.name, None
            if item.get("url"):
                # download url
                file_bytes = await async_searcher(item.get("url"), re_content=True)
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tf.write(file_bytes)
                tf.close()
                return tf.name, None
        return None, json.dumps(data)
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error parsing image response: {e}"


async def _image_edit(image_path: str, prompt: str):
    api_key = get_api_key()
    if not api_key:
        return None, "⚠️ AI API key not set. Set it using `setdb AI_API_KEY your_api_key` or export `AI_API_KEY` in your .env."
    base = get_api_base()
    url = f"{base}/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with aiohttp.ClientSession() as s:
            with open(image_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("image", f, filename="image.png", content_type="image/png")
                data.add_field("prompt", prompt)
                data.add_field("n", "1")
                async with s.post(url, data=data, headers=headers, timeout=120) as resp:
                    res = await resp.json()
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error contacting image edit API: {e}"

    try:
        item = res.get("data", [None])[0]
        if not item:
            return None, json.dumps(res)
        if item.get("b64_json"):
            b = base64.b64decode(item["b64_json"])
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tf.write(b)
            tf.close()
            return tf.name, None
        if item.get("url"):
            file_bytes = await async_searcher(item.get("url"), re_content=True)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tf.write(file_bytes)
            tf.close()
            return tf.name, None
        return None, json.dumps(res)
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error parsing image edit response: {e}"


async def _transcribe_audio(audio_path: str):
    api_key = get_api_key()
    if not api_key:
        return None, "⚠️ AI API key not set. Set it using `setdb AI_API_KEY your_api_key` or export `AI_API_KEY` in your .env."
    base = get_api_base()
    url = f"{base}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with aiohttp.ClientSession() as s:
            with open(audio_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="audio.ogg", content_type="audio/ogg")
                data.add_field("model", "whisper-1")
                async with s.post(url, data=data, headers=headers, timeout=180) as resp:
                    res = await resp.json()
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error contacting transcription API: {e}"

    try:
        # OpenAI returns {text:...}
        if isinstance(res, dict) and res.get("text"):
            return res.get("text"), None
        # other shapes may use 'transcript' or choices
        return None, json.dumps(res)
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error parsing transcription response: {e}"


@ultroid_cmd(pattern="ai( (.*)|$)")
async def ai_handler(event):
    """AI command: chat queries, image create/edit, transcribe, reply analysis"""
    arg = event.pattern_match.group(1).strip()
    reply = await event.get_reply_message()
    msg = await eor(event, "🤖 Processing...")

    # Image creation
    if arg.lower().startswith("create"):
        prompt = arg[len("create"):].strip()
        if not prompt:
            return await msg.edit("❌ Provide a prompt for image creation, e.g. `.ai create a red apple`")
        file_path, err = await _image_create(prompt)
        if err:
            return await msg.edit(err)
        await event.client.send_file(event.chat_id, file_path, reply_to=event.reply_to_msg_id)
        await msg.delete()
        try:
            os.remove(file_path)
        except Exception:
            pass
        return

    # If replied to a message
    if reply:
        # Image edit (replying to image/sticker/photo)
        if reply.photo or (hasattr(reply, "media") and getattr(reply, "media") and getattr(reply, "media").__class__.__name__.endswith("Document") and getattr(reply, "file") is not None and getattr(reply.file, "mime_type", "").startswith("image")):
            if not arg:
                return await msg.edit("❌ Provide edit instructions after `.ai`, e.g. `.ai make it look like a painting`")
            img = await reply.download_media()
            file_path, err = await _image_edit(img, arg)
            try:
                os.remove(img)
            except Exception:
                pass
            if err:
                return await msg.edit(err)
            await event.client.send_file(event.chat_id, file_path, reply_to=event.reply_to_msg_id)
            await msg.delete()
            try:
                os.remove(file_path)
            except Exception:
                pass
            return

        # Audio/voice reply -> transcribe then optionally answer/summarize
        if reply.voice or (reply.media and getattr(reply.file, "mime_type", "").startswith("audio")):
            tf = await reply.download_media()
            transcript, err = await _transcribe_audio(tf)
            try:
                os.remove(tf)
            except Exception:
                pass
            if err:
                return await msg.edit(err)
            # If user asked for transcript explicitly
            if not arg or "transcript" in arg.lower() or "transcribe" in arg.lower() or arg.lower().startswith("give transcript"):
                await event.reply(f"📝 Transcript:\n{transcript}")
                await msg.delete()
                return
            # Otherwise treat arg as question/instruction about audio
            combined = f"Transcript:\n{transcript}\n\nUser instruction: {arg}"
            resp = await _chat_completion(combined)
            await event.reply(resp)
            await msg.delete()
            return

        # Text reply -> analyze / suggest reply
        if reply.message:
            original = reply.message
            if not arg:
                # suggest short reply
                prompt = f"You are a helpful assistant. Suggest a short, polite reply to the following message:\n\n{original}\n\nReply:" 
            else:
                prompt = f"Analyze the following message and {arg}:\n\n{original}" 
            resp = await _chat_completion(prompt)
            await event.reply(resp)
            await msg.delete()
            return

    # Fallback: plain chat query
    if not arg:
        return await msg.edit("❌ Please provide a question or use `.ai create <prompt>` for images or reply to media with `.ai <instruction>`.")
    resp = await _chat_completion(arg)
    await msg.edit(resp)
