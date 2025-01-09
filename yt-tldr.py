import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
import re

from openai import OpenAI, AsyncOpenAI
import whisper
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Language support configuration
SUPPORTED_LANGUAGES: Dict[str, str] = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'id': 'Indonesian',
    'hi': 'Hindi',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'tr': 'Turkish',
    'ru': 'Russian',
    'vi': 'Vietnamese',
    'th': 'Thai'
}

@dataclass
class Config:
    """Configuration class to store all settings"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    temp_dir: Path = Path("temp")
    output_dir: Path = Path("output")
    supported_whisper_models: tuple = ("base", "small", "medium", "large")
    max_retries: int = 3
    timeout: int = 300  # 5 minutes

class YouTubeSummarizer:
    def __init__(self, config: Config):
        self.config = config
        self.validate_config()
        
        # Create necessary directories
        self.config.temp_dir.mkdir(exist_ok=True)
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.async_openai_client = AsyncOpenAI(api_key=config.openai_api_key)

    def validate_config(self) -> None:
        """Validate configuration settings"""
        if not self.config.openai_api_key:
            raise ValueError("Missing OpenAI API key in .env file")

    async def get_video_info(self, youtube_url: str) -> Optional[Tuple[str, str]]:
        """Gets video ID and title from YouTube URL"""
        try:
            command = [
                "yt-dlp",
                "--get-id",
                "--get-title",
                youtube_url
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Failed to get video info: {stderr.decode()}")
                return None
                
            # Split output into lines and get video ID and title
            output_lines = stdout.decode().strip().split('\n')
            if len(output_lines) >= 2:
                video_title = output_lines[0]
                video_id = output_lines[1]
                return video_id, video_title
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    def sanitize_filename(self, filename: str) -> str:
        """Sanitizes the filename by removing invalid characters"""
        # Remove or replace invalid filename characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit filename length
        return filename[:100]

    async def download_audio(self, youtube_url: str) -> Optional[Path]:
        """Downloads audio from YouTube URL"""
        audio_file = self.config.temp_dir / "temp_audio.mp3"
        
        try:
            command = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "mp3",
                "--output", str(audio_file),
                youtube_url
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Download failed: {stderr.decode()}")
                return None
                
            return audio_file
            
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return None

    async def transcribe_audio(
        self, 
        audio_file: Path, 
        model_name: str = "base", 
        language: str = "en"
    ) -> Optional[str]:
        """Transcribes audio using Whisper with language support"""
        if model_name not in self.config.supported_whisper_models:
            raise ValueError(f"Unsupported model. Choose from: {self.config.supported_whisper_models}")
            
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language. Choose from: {', '.join(SUPPORTED_LANGUAGES.keys())}")
            
        try:
            logger.info(f"Loading Whisper model: {model_name}")
            model = whisper.load_model(model_name)
            
            logger.info(f"Starting transcription in {SUPPORTED_LANGUAGES[language]}...")
            result = model.transcribe(
                str(audio_file),
                language=language,
                fp16=False  # Avoid GPU memory issues
            )
            
            logger.info("Transcription completed")
            return result["text"]
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def preprocess_transcript(self, text: str) -> str:
        """Preprocess the transcript for better summarization"""
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove timestamps if they exist in the format [00:00:00]
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        
        # Remove speaker labels if they exist (e.g., "Speaker 1:", "John:")
        text = re.sub(r'\b[A-Za-z]+\s*\d*\s*:', '', text)
        
        # Split into sentences for better processing
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove very short sentences (likely noise)
        sentences = [s for s in sentences if len(s.split()) > 3]
        
        return ' '.join(sentences)

    async def summarize_text(self, text: str, language: str = "en") -> Optional[str]:
        """Enhanced text summarization with better prompting and chunking"""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language. Choose from: {', '.join(SUPPORTED_LANGUAGES.keys())}")

        # Preprocess the transcript
        processed_text = self.preprocess_transcript(text)
        
        # Split text into chunks if it's too long (8000 tokens approximate)
        chunks = self._split_text_into_chunks(processed_text, chunk_size=8000)
        summaries = []

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Summarizing chunk {i} of {len(chunks)}...")
            chunk_summary = await self._summarize_chunk(chunk, language)
            if chunk_summary:
                summaries.append(chunk_summary)

        if not summaries:
            return None

        # If we have multiple summaries, combine them
        if len(summaries) > 1:
            logger.info("Creating final combined summary...")
            combined_summary = "\n\n".join(summaries)
            return await self._create_final_summary(combined_summary, language)
        
        return summaries[0]

    async def _summarize_chunk(self, text: str, language: str) -> Optional[str]:
        """Summarize a single chunk of text"""
        prompt = self._create_summary_prompt(text, language)
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.async_openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""You are an expert content summarizer. 
                            Create a detailed, well-structured summary in {SUPPORTED_LANGUAGES[language]}.
                            Focus on extracting key insights, main arguments, and important details.
                            Use clear headings and bullet points for better readability.
                            Be comprehensive yet concise, and maintain the original meaning and nuance."""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more focused output
                    max_tokens=2000,  # Increased token limit
                    timeout=self.config.timeout
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error("Max retries reached for summarization")
                    return None
                await asyncio.sleep(2 ** attempt)

    async def _create_final_summary(self, combined_summaries: str, language: str) -> Optional[str]:
        """Create a final, coherent summary from multiple chunk summaries"""
        try:
            response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""Create a cohesive, well-structured final summary in {SUPPORTED_LANGUAGES[language]} 
                        from the following combined summaries. Ensure the final summary is comprehensive, 
                        well-organized, and maintains a clear narrative flow."""
                    },
                    {"role": "user", "content": combined_summaries}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error creating final summary: {e}")
            return combined_summaries

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> list:
        """Split text into chunks based on approximate token count"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word.split())
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word.split())
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _create_summary_prompt(self, text: str, language: str) -> str:
        """Enhanced prompt for better summaries"""
        prompts = {
            "en": """
                Please provide a comprehensive analysis and summary of the following content.
                
                Focus on:
                1. Main Topics and Key Points
                   - Core arguments and central themes
                   - Key findings or discoveries
                   - Critical insights and observations
                
                2. Detailed Analysis
                   - Supporting evidence and examples
                   - Statistical data or research mentioned
                   - Expert opinions or citations
                
                3. Context and Implications
                   - Background information
                   - Broader impact and significance
                   - Future implications or recommendations
                
                4. Notable Elements
                   - Significant quotes or statements
                   - Unique perspectives or approaches
                   - Controversial or debated points
                
                Please structure the summary with clear headings, subheadings, and bullet points.
                Maintain the depth and nuance of the original content while presenting it in a clear,
                organized format.
                
                Text to analyze and summarize:
            """,
            # Add other language prompts as needed
        }
        
        prompt_template = prompts.get(language, prompts["en"])
        return f"{prompt_template}\n\n{text}"

    def generate_mdx(self, video_id: str, transcription: str, summary: str, language: str) -> str:
        """Generates enhanced MDX content with metadata and language information"""
        return f"""---
title: "YouTube Video Summary"
videoId: "{video_id}"
language: "{SUPPORTED_LANGUAGES[language]}"
date: "{datetime.now().isoformat()}"
---

# Transcription

{transcription}

---

# Detailed Summary

{summary}

---

*Generated using YouTube Summarizer v1.0*
"""

async def main():
    config = Config()
    summarizer = YouTubeSummarizer(config)
    
    # Print available languages
    print("\nAvailable languages:")
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"{code}: {name}")
        
    youtube_url = input("\nEnter YouTube URL: ").strip()
    model_choice = input(f"Choose Whisper model {config.supported_whisper_models}: ").strip()
    language_choice = input("Choose language code (e.g., en, es, fr): ").strip().lower()
    
    if language_choice not in SUPPORTED_LANGUAGES:
        logger.error(f"Unsupported language code. Using English (en) as default.")
        language_choice = "en"
    
    try:
        # Get video info first
        video_info = await summarizer.get_video_info(youtube_url)
        if not video_info:
            logger.error("Failed to get video information")
            return
            
        video_id, video_title = video_info
        
        # Download audio
        logger.info("Downloading audio...")
        audio_file = await summarizer.download_audio(youtube_url)
        if not audio_file:
            logger.error("Audio download failed")
            return

        # Transcribe
        logger.info(f"Starting transcription in {SUPPORTED_LANGUAGES[language_choice]}...")
        transcription = await summarizer.transcribe_audio(audio_file, model_choice, language_choice)
        if not transcription:
            logger.error("Transcription failed")
            return

        # Summarize
        logger.info(f"Generating summary in {SUPPORTED_LANGUAGES[language_choice]}...")
        summary = await summarizer.summarize_text(transcription, language_choice)
        if not summary:
            logger.error("Summarization failed")
            return

        # Generate and save MDX
        mdx_content = summarizer.generate_mdx(video_id, transcription, summary, language_choice)
        
        # Create filename using sanitized video title
        sanitized_title = summarizer.sanitize_filename(video_title)
        output_file = config.output_dir / f"{sanitized_title}_{language_choice}.mdx"
        
        output_file.write_text(mdx_content, encoding="utf-8")
        logger.info(f"MDX content saved to {output_file}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        # Cleanup
        for file in config.temp_dir.glob("*"):
            file.unlink()

if __name__ == "__main__":
    asyncio.run(main())