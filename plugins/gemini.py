import base64
import mimetypes
import os

from . import (
    CMD_HELP,
    LOGS,
    eod,
    eor,
    get_string,
    udB,
    ultroid_cmd,
    async_searcher,
)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}"
DEFAULT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"


@ultroid_cmd(pattern="gem")
async def gemini(event):
    """
    Command to get response from Google Gemini.
    Usage:
    .gem <prompt>
    .gem <reply to a message> <prompt>
    .gem <reply to a media> <prompt>
    """
    msg = await eor(event, get_string("processing"))
    
    api_key = udB.get_key("GEMINI_API_KEY")
    if not api_key:
        error_message = (
            "⚠️ Please set Gemini API key using `setdb GEMINI_API_KEY your_api_key`\n"
            "You can also set a custom model using `setdb GEMINI_MODEL gemini-1.5-flash`."
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
                return await eod(msg, get_string("media_upload_error"), time=10)

        if reply_msg and reply_msg.text:
            prompt = f"{reply_msg.text}\n\n{prompt}" if prompt else reply_msg.text

    if not prompt and not media_data:
        return await eod(msg, get_string("no_prompt"), time=10)

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
            if error.get("code") == 429:
                error_msg = "⚠️ Rate limit exceeded. Please try again later."
            return await eod(msg, error_msg, time=10)

        if "promptFeedback" in response:
            feedback = response["promptFeedback"]
            if feedback.get("blockReason"):
                return await eod(msg, f"Blocked for: {feedback['blockReason']}", time=10)

        if "candidates" not in response or not response["candidates"]:
             return await eod(msg, "No response from Gemini.", time=10)

        result = ""
        for part in response["candidates"][0]["content"]["parts"]:
            if "text" in part:
                result += part["text"]

        if not result:
            return await eod(msg, "Empty response from Gemini.", time=10)

        if len(result) > 4096:
            await eor(msg, result[:4096])
            result = result[4096:]
            for i in range(0, len(result), 4096):
                await event.reply(result[i:i+4096])
        else:
            await eor(msg, result)
        
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
