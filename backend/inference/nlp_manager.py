import os
import google.generativeai as genai
from typing import Optional

class LLMProcessor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-flash-latest')
                print("LLMProcessor: Gemini model initialized successfully.")
            except Exception as e:
                print(f"LLMProcessor: Error initializing Gemini: {e}")
        else:
            print("LLMProcessor: No API key found. Falling back to dictionary mode.")

    def correct_sentence(self, raw_voice_text: str, audio_file_path: Optional[str] = None) -> str:
        """
        Uses LLM to fuse raw voice-reading characters and actual audio into perfect English.
        """
        if not self.model:
            return raw_voice_text
            
        target_words = len(raw_voice_text.split())

        # STRICT TARGET DICTIONARY FOR LLM
        TARGET_LIST = [
            "HELLO", "HI", "GOOD MORNING", "HOW ARE YOU", "WHAT IS YOUR NAME", "MY NAME IS", 
            "THANK YOU", "WELCOME", "YES", "NO", "RAYYAN", "ANGEL", 
            "ARDRA", "NITHYA", "SUJITHRA", "RENJINI"
        ]
        target_str = ", ".join(TARGET_LIST)

        # Base Instructions
        instructions = f"""
        You are a Robotic Text Echo Script. You have NO intelligence.
        The user has spoken, and you have two inputs:
        1. RAW VOICE DATA: "{raw_voice_text}"
        2. AUDIO FILE: (Recording of the voice)
        
        STRICT PROTOCOL:
        1. TARGET DICTIONARY LIMIT: You may ONLY output a phrase EXACTLY matching one from this list: [{target_str}].
        2. NO REPLIES: You are a mirror. If the user says "HELLO", you output "HELLO".
        3. GIBBERISH REJECTION: If the RAW VOICE DATA or AUDIO clearly does NOT match ANY phrase in the TARGET DICTIONARY, you MUST output an EMPTY STRING.
        4. OUPUT: Return ONLY the exact phrase from the dictionary in UPPERCASE. DO NOT include prefixes. DO NOT be conversational.
        """

        try:
            content = [instructions]
            
            # If audio file provided, upload and attach to prompt
            if audio_file_path and os.path.exists(audio_file_path) and os.path.getsize(audio_file_path) > 1000:
                audio_file = genai.upload_file(path=audio_file_path)
                content.append(audio_file)
            else:
                content.append(f"RAW VOICE DATA: \"{raw_voice_text}\"")

            config = genai.GenerationConfig(temperature=0.0)
            response = self.model.generate_content(content, generation_config=config)
            result = response.text.strip().upper()
            
            # Post-generation strict cleanup
            if result.startswith("I HEARD:") or result.startswith("OUTPUT:"):
                result = result.split(":", 1)[-1].strip()
                
            print(f"DEBUG: LLM Correction Result: '{result}' (from raw: '{raw_voice_text}')")
            return result
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                print("LLMProcessor: QUOTA EXCEEDED. Returning empty for safety.")
                return "" # Return nothing rather than mess
            return "" # Final safety: return nothing rather than mess
