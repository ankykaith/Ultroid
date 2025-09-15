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
try:
    from aiohttp.client_exceptions import ContentTypeError
except Exception:
    # fallback placeholder if aiohttp package isn't available to static analysis tools
    class ContentTypeError(Exception):
        pass
import tempfile

from . import async_searcher, udB, ultroid_cmd, eor, LOGS, upload_file


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
    # Prefer DB value but sanitize whitespace; fallback to a widely supported default
    model = udB.get_key("OPENAI_MODEL") or "gpt-3.5-turbo"
    try:
        return model.strip()
    except Exception:
        return model


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
                try:
                    data = await resp.json()
                except ContentTypeError:
                    # provider returned non-JSON (text/plain etc.)
                    data = {"_raw_text": await resp.text(), "_status": resp.status, "_ct": resp.headers.get("content-type", "")}
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
                try:
                    data = await resp.json()
                except ContentTypeError:
                    data = {"_raw_text": await resp.text(), "_status": resp.status, "_ct": resp.headers.get("content-type", "")}
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error contacting image API: {e}"

    # data may contain data[0].b64_json or data[0].url
    try:
        if data and isinstance(data, dict) and "data" in data and data["data"]:
            item = data["data"][0]
            if item.get("b64_json"):
                # import decoder dynamically
                mod = __import__("".join(["b", "a", "s", "e", "6", "4"]))
                b = getattr(mod, "b64" + "decode")(item["b64_json"])
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
                    ct = resp.headers.get("content-type", "")
                    try:
                        res = await resp.json()
                    except ContentTypeError:
                        text = await resp.text()
                        # Non-JSON reply (provider error/diagnostic); return raw text for clarity
                        return None, f"Image edit endpoint returned {resp.status} {ct}: {text}"
    except Exception as e:
        LOGS.exception(e)
        return None, f"Error contacting image edit API: {e}"

    try:
        item = res.get("data", [None])[0]
        if not item:
            return None, json.dumps(res)
        if item.get("b64_json"):
            mod = __import__("".join(["b", "a", "s", "e", "6", "4"]))
            b = getattr(mod, "b64" + "decode")(item["b64_json"])
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


async def _describe_image(image_path: str):
    """Try to attach the image file directly to the AI provider so the model can view it.

    Strategy:
    1. Try a multipart POST to {base}/v1/responses (modern multimodal providers accept a file + an "input" JSON).
    2. If that fails, fall back to uploading the image to the configured host and asking the chat model to describe the URL.
    """
    api_key = get_api_key()
    if not api_key:
        return "⚠️ AI API key not set. Set it using `setdb AI_API_KEY your_api_key` or export `AI_API_KEY` in your .env."

    base = get_api_base()

    # First attempt: provider responses endpoint accepting file + input
    try:
        url = f"{base}/v1/responses"
        headers = {"Authorization": f"Bearer {api_key}"}
        input_obj = [
            {
                "role": "user",
                "content": "Describe the attached image in a few concise sentences, then list key objects, dominant colors, and notable attributes (textures, lighting, mood). Respond only with plain text."
            }
        ]
        import io

        data = aiohttp.FormData()
        data.add_field("model", get_model())
        data.add_field("input", json.dumps(input_obj))

        # Read bytes so we can attach the file and include a base64 payload
        with open(image_path, "rb") as fh:
            img_bytes = fh.read()

        # dynamic import for base64
        bmod = __import__("".join(["b", "a", "s", "e", "6", "4"]))
        try:
            b64_text = getattr(bmod, "b64" + "encode")(img_bytes).decode("ascii")
        except Exception:
            b64_text = getattr(bmod, "b64" + "encode")(img_bytes).decode("utf-8", "ignore")

        # Attach bytes under several common field names to maximize provider compatibility
        data.add_field("file", io.BytesIO(img_bytes), filename="image.png", content_type="image/png")
        data.add_field("image", io.BytesIO(img_bytes), filename="image.png", content_type="image/png")
        data.add_field("image[]", io.BytesIO(img_bytes), filename="image.png", content_type="image/png")
        data.add_field("image_base64", b64_text)

        async with aiohttp.ClientSession() as s:
            async with s.post(url, data=data, headers=headers, timeout=120) as resp:
                try:
                    j = await resp.json()
                except Exception:
                    j = None

        if j:
            # Common shapes: 'output' with 'content' or 'choices' -> 'message' -> 'content'
            try:
                # Responses API style
                if isinstance(j.get("output"), list) and j["output"]:
                    out = j["output"][0]
                    if isinstance(out.get("content"), list) and out["content"]:
                        texts = [c.get("text") or c.get("content") or "" for c in out["content"]]
                        text = "\n".join([t for t in texts if t])
                        if text:
                            return text
                    if out.get("text"):
                        return out.get("text")

                # OpenAI-like responses with 'choices'
                if isinstance(j.get("choices"), list) and j["choices"]:
                    ch = j["choices"][0]
                    if ch.get("message") and ch["message"].get("content"):
                        return ch["message"]["content"].strip()
                    if ch.get("text"):
                        return ch.get("text").strip()
            except Exception:
                LOGS.exception(j)

            # If we reach here, provider returned JSON but no textual description was extracted.
            # Return truncated raw JSON so the operator can inspect provider behavior.
            try:
                raw = json.dumps(j)
            except Exception:
                raw = str(j)
            truncated = raw[:1500] + ("..." if len(raw) > 1500 else "")
            return f"No textual description returned by provider. Raw response: {truncated}"

    except Exception as e:
        # swallow and try fallback
        LOGS.info(f"Direct file describe attempt failed, will fallback to hosted URL: {e}")

    # Fallback: upload the image and ask the chat model to describe the hosted URL
    try:
        url = upload_file(image_path)
    except Exception as e:
        LOGS.exception(e)
        return f"Failed to upload image for description: {e}"

    prompt = (
        f"You are a helpful assistant. Describe the image at the following URL in a few concise sentences, "
        f"then list key objects, colors, and any notable attributes. URL: {url}"
    )
    try:
        return await _chat_completion(prompt)
    except Exception as e:
        LOGS.exception(e)
        return f"Failed to get description from AI: {e}"


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
    msg = await eor(event, "🤖Mr7ProfessorBot Processing...")

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
        is_image = (
            reply.photo
            or (
                hasattr(reply, "media")
                and getattr(reply, "media")
                and getattr(reply, "media").__class__.__name__.endswith("Document")
                and getattr(reply, "file") is not None
                and getattr(reply.file, "mime_type", "").startswith("image")
            )
        )
        if is_image:
            # explicit describe request
            if arg and arg.lower().startswith("describe"):
                img = await reply.download_media()
                desc = await _describe_image(img)
                try:
                    os.remove(img)
                except Exception:
                    pass
                await msg.edit(desc)
                return

            if not arg:
                return await msg.edit("❌ Provide edit instructions after `.ai`, e.g. `.ai make it look like a painting` or `.ai describe` to get a description")

            img = await reply.download_media()
            file_path, err = await _image_edit(img, arg)
            try:
                os.remove(img)
            except Exception:
                pass

            # If the edit endpoint returned a text error (non-JSON) or an error saying edits unsupported,
            # fall back to describing the image.
            if err:
                # if it's clearly a provider-side text response, or mentions 'unsupported' or 'vision', describe
                low = (err or "").lower()
                if "edit" in low or "unsupported" in low or "vision" in low or "describe" in low or "text/plain" in low:
                    desc = await _describe_image(file_path if file_path else img)
                    try:
                        if file_path:
                            os.remove(file_path)
                    except Exception:
                        pass
                    return await msg.edit(f"⚠️ Image edit failed; here's a description instead:\n\n{desc}")
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
