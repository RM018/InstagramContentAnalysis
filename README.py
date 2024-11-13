# InstagramContentAnalysis

# backend/content_analysis/instagram_scraper.py
import instaloader
from moviepy.editor import VideoFileClip
import os
from typing import List, Dict
import asyncio
import logging

class InstagramScraper:
    def __init__(self):
        """Initialize Instagram scraper with configuration"""
        self.loader = instaloader.Instaloader(
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=True
        )
        self.temp_dir = "temp_downloads"
        os.makedirs(self.temp_dir, exist_ok=True)

    async def authenticate(self, username: str, password: str) -> bool:
        """Authenticate with Instagram"""
        try:
            self.loader.login(username, password)
            return True
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            return False

    async def fetch_creator_content(self, username: str, max_posts: int = 10) -> Dict:
        """Fetch recent videos from creator's profile"""
        try:
            profile = instaloader.Profile.from_username(self.loader.context, username)
            posts = profile.get_posts()
            
            videos = []
            captions = []
            
            for post in posts:
                if len(videos) >= max_posts:
                    break
                    
                if post.is_video:
                    video_data = await self._process_video(post)
                    if video_data:
                        videos.append(video_data)
                        captions.append(post.caption or "")
            
            return {
                'videos': videos,
                'captions': captions,
                'profile_info': {
                    'username': username,
                    'followers': profile.followers,
                    'posts_count': profile.mediacount
                }
            }
            
        except Exception as e:
            logging.error(f"Content fetching failed: {e}")
            return None

    async def _process_video(self, post) -> Dict:
        """Process individual video and extract features"""
        try:
            video_path = f"{self.temp_dir}/{post.shortcode}.mp4"
            self.loader.download_post(post, target=self.temp_dir)
            
            with VideoFileClip(video_path) as video:
                audio_path = await self._extract_audio(video, post.shortcode)
                
                return {
                    'shortcode': post.shortcode,
                    'duration': video.duration,
                    'audio_path': audio_path,
                    'timestamp': post.date_local,
                    'likes': post.likes,
                    'views': post.video_view_count
                }
                
        except Exception as e:
            logging.error(f"Video processing failed: {e}")
            return None
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    async def _extract_audio(self, video: VideoFileClip, shortcode: str) -> str:
        """Extract audio from video clip"""
        audio_path = f"{self.temp_dir}/audio_{shortcode}.mp3"
        try:
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            logging.error(f"Audio extraction failed: {e}")
            return None


# backend/content_analysis/style_analyzer.py
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict
import spacy
from collections import Counter
import requests
from bs4 import BeautifulSoup

class StyleAnalyzer:
    def __init__(self):
        """Initialize style analysis components"""
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.speech_recognizer = pipeline("automatic-speech-recognition")

    async def analyze_creator_style(self, audio_files: List[str], captions: List[str]) -> Dict:
        """Analyze creator's content style"""
        try:
            # Process audio and text in parallel
            audio_features = await self._analyze_audio(audio_files)
            text_features = await self._analyze_text(captions)
            
            return {
                'audio_profile': audio_features,
                'text_profile': text_features,
                'combined_style': self._combine_profiles(audio_features, text_features)
            }
        except Exception as e:
            logging.error(f"Style analysis failed: {e}")
            return None

    async def _analyze_audio(self, audio_files: List[str]) -> Dict:
        """Analyze audio characteristics"""
        transcripts = []
        speech_patterns = []
        
        for audio_file in audio_files:
            transcript = self.speech_recognizer(audio_file)
            transcripts.append(transcript['text'])
            patterns = await self._analyze_speech_patterns(audio_file)
            speech_patterns.append(patterns)
        
        return {
            'transcripts': transcripts,
            'speech_patterns': speech_patterns,
            'common_phrases': self._extract_common_phrases(transcripts),
            'speech_stats': self._calculate_speech_stats(speech_patterns)
        }

    async def _analyze_speech_patterns(self, audio_file: str) -> Dict:
        # TODO: Implement detailed speech pattern analysis
        pass

    async def _analyze_text(self, captions: List[str]) -> Dict:
        processed_texts = [self.nlp(text) for text in captions]
        return {
            'common_phrases': self._extract_common_phrases(captions),
            'sentiment_profile': self._analyze_sentiment(captions),
            'language_stats': self._calculate_language_stats(processed_texts)
        }

    def _extract_common_phrases(self, texts: List[str]) -> List[str]:
        docs = [self.nlp(text) for text in texts]
        phrases = []
        
        for doc in docs:
            phrases.extend([chunk.text for chunk in doc.noun_chunks])
            phrases.extend([token.text for token in doc if token.dep_ == "ROOT"])
        
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(10)]

    def _analyze_sentiment(self, texts: List[str]) -> Dict:
        sentiments = [self.sentiment_analyzer(text)[0] for text in texts]
        return {
            'average_sentiment': np.mean([s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiments]),
            'sentiment_distribution': Counter(s['label'] for s in sentiments)
        }

    def _calculate_language_stats(self, processed_texts: List) -> Dict:
        stats = {
            'avg_sentence_length': np.mean([len(text) for text in processed_texts]),
            'vocabulary_size': len(set(token.text for text in processed_texts for token in text)),
            'pos_distribution': Counter(token.pos_ for text in processed_texts for token in text)
        }
        return stats

    def _combine_profiles(self, audio_profile: Dict, text_profile: Dict) -> Dict:
        return {
            'speaking_style': {
                'pace': audio_profile['speech_stats']['average_pace'],
                'patterns': audio_profile['speech_patterns']
            },
            'language_style': {
                'common_phrases': text_profile['common_phrases'],
                'sentiment': text_profile['sentiment_profile']
            },
            'content_patterns': {
                'structure': self._analyze_content_structure(audio_profile['transcripts'], text_profile['language_stats'])
            }
        }

# Product Review Generation
class ProductReviewGenerator:
    def __init__(self, style_text_generator):
        self.style_text_generator = style_text_generator

    def fetch_product_details(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find("title").get_text()
        description = soup.find("meta", {"name": "description"})["content"]
        return f"Product: {title}\nDescription: {description}"

    def generate_review(self, url: str) -> str:
        product_info = self.fetch_product_details(url)
        return self.style_text_generator.generate_text(f"Review this product in my style: {product_info}")

# Video Script Generator
class VideoScriptGenerator:
    def __init__(self, style_text_generator):
        self.style_text_generator = style_text_generator
    
    def create_script(self, topic: str) -> str:
        prompt = f"Generate a video script about {topic} in my style."
        return self.style_text_generator.generate_text(prompt)

# Voice Synthesis
class VoiceSynthesizer:
    def __init__(self, trained_voice_model_path):
        self.tts_model = tts.load_model(trained_voice_model_path)

    def synthesize_voice(self, text: str, output_path: str):
        audio = self.tts_model.synthesize(text)
        with open(output_path, "wb") as f:
            f.write(audio)
