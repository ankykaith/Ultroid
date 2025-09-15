
import base64
import mimetypes
import os
import json

from . import (
    CMD_HELP,
    LOGS,
    eod,
    eor,
    udB,
    ultroid_cmd,
    async_searcher,
)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}"
DEFAULT_MODEL = "gemini-pro"
VISION_MODEL = "gemini-pro-vision"

@ultroid_cmd(pattern="gem")
async def gemini(event):
    """
    Command to get response from Google Gemini.
    Usage:
    .gem <prompt>
    .gem <reply to a message> <prompt>
    .gem <reply to a media> <prompt>
    """
    msg = await eor(event, "Processing...")
    
    api_key = udB.get_key("GEMINI_API_KEY")
    if not api_key:
        error_message = (
            "⚠️ Please set Gemini API key using `setdb GEMINI_API_KEY your_api_key`\n"
            "You can also set a custom model using `setdb GEMINI_MODEL <model_name>`."
        )
        return await eod(msg, error_message, time=10)
    
    prompt = event.text.split(maxsplit=1)[1] if event.text.split(maxsplit=1)[1:] else ""
    media_data = None
    
    if event.reply_to_msg_id:
        reply_msg = await event.get_reply_message()
        
        if reply_msg and reply_msg.media:
            try:
                file_path = await event.client.download_media(reply_msg.media)
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type:
                    with open(file_path, "rb") as f:
                        file_content = base64.b64encode(f.read()).decode("utf-8")
                    media_data = {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": file_content,
                        }
                    }
                os.remove(file_path)
            except Exception as e:
                LOGS.exception(e)
                return await eod(msg, "An error occurred while uploading the media.", time=10)

        if reply_msg and reply_msg.text:
            prompt = f"{reply_msg.text}\n\n{prompt}" if prompt else reply_msg.text

    if not prompt and not media_data:
        return await eod(msg, "Please provide a prompt.", time=10)

    model = udB.get_key("GEMINI_MODEL")
    if media_data:
        model = model or VISION_MODEL
    else:
        model = model or DEFAULT_MODEL

    parts = []
    if prompt:
        parts.append({"text": prompt})
    if media_data:
        parts.append(media_data)

    try:
        response = await async_searcher(
            API_URL.format(model, api_key),
            post=True,
            json={"contents": [{"parts": parts}]},
            re_json=True,
        )
        
        if "error" in response:
            error = response["error"]
            error_msg = f"Error: {error.get('message', 'Unknown error occurred')}"
            return await eod(msg, error_msg, time=10)

        if "promptFeedback" in response and response["promptFeedback"].get("blockReason"):
            return await eod(msg, f"Blocked for: {response['promptFeedback']['blockReason']}", time=10)

        if not response.get("candidates"):
            LOGS.info(f"Gemini Response: {json.dumps(response, indent=2)}")
            return await eod(msg, "Request was blocked or the model could not generate a response. Check logs for details.", time=10)

        result_text = ""
        media_path = None
        
        candidate = response["candidates"][0]
        
        if candidate.get("finishReason") == "SAFETY":
            ratings = candidate.get("safetyRatings", [])
            reasons = [r['category'] for r in ratings if r['probability'] != 'NEGLIGIBLE']
            return await eod(msg, f"Blocked for safety reasons: {', '.join(reasons)}", time=10)

        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            if "text" in part:
                result_text += part["text"]
            if "fileData" in part:
                file_data = part["fileData"]
                uri = file_data.get("fileUri", "")
                if uri.startswith("data:") and ";base64," in uri:
                    _, encoded = uri.split(",", 1)
                    mime_type = file_data.get("mimeType")
                    file_content = base64.b64decode(encoded)
                    
                    extension = mimetypes.guess_extension(mime_type) or ".out"
                    media_path = f"gemini_media{extension}"
                    with open(media_path, "wb") as f:
                        f.write(file_content)

        if media_path:
            await msg.delete()
            await event.reply(file=media_path, message=result_text or None)
            os.remove(media_path)
            return

        if not result_text.strip():
            LOGS.info(f"Gemini Response: {json.dumps(response, indent=2)}")
            return await eod(msg, "Empty response from Gemini. The model may not support this type of request. Check logs for details.", time=10)

        if len(result_text) > 4096:
            await eor(msg, result_text[:4096])
            result_text = result_text[4096:]
            for i in range(0, len(result_text), 4096):
                await event.reply(result_text[i:i+4096])
        else:
            await eor(msg, result_text)
        
    except Exception as e:
        LOGS.exception(e)
        await eod(msg, f"An unexpected error occurred: {str(e)}", time=10)

CMD_HELP.update(
    {
        "gemini": {
            "command": "gem",
            "usage": "<prompt> or <reply to a message/media> <prompt>",
            "description": "Get response from Google Gemini.",
        }
    }
)
