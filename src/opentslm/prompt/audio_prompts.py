# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Audio-specific prompts for audio language models.

This module provides prompt templates for various audio understanding tasks
including speech recognition, audio captioning, sound classification, and more.
"""

# Speech Recognition Prompts
SPEECH_RECOGNITION_PROMPTS = [
    "Listen to the following speech audio and transcribe what is said.",
    "Transcribe the speech in this audio clip.",
    "What words are spoken in this audio?",
    "Please provide a transcription of the speech audio.",
    "Convert the following speech to text.",
]

# Audio Captioning Prompts
AUDIO_CAPTIONING_PROMPTS = [
    "Listen to the following audio clip and describe what you hear.",
    "Describe the sounds in this audio recording.",
    "What sounds can you identify in this audio?",
    "Provide a detailed description of the audio content.",
    "Caption this audio clip by describing all the sounds you hear.",
]

# Sound Classification Prompts
SOUND_CLASSIFICATION_PROMPTS = [
    "What type of sound is this? Choose from: {options}",
    "Classify this audio into one of the following categories: {options}",
    "Listen to the audio and identify which category it belongs to: {options}",
    "Which of these best describes the sound: {options}?",
]

# Audio Question Answering Prompts
AUDIO_QA_PROMPTS = [
    "Listen to the audio and answer the following question: {question}",
    "Based on the audio clip, {question}",
    "After listening to the audio, please answer: {question}",
]

# Music Understanding Prompts
MUSIC_ANALYSIS_PROMPTS = [
    "Analyze this music and describe the genre, instruments, and mood.",
    "What musical instruments can you hear in this audio?",
    "Describe the genre and style of this music.",
    "What is the mood or emotion conveyed by this music?",
    "Provide a detailed analysis of this musical piece.",
]

# Speaker Identification Prompts
SPEAKER_ID_PROMPTS = [
    "How many speakers are in this audio?",
    "Identify the number of different speakers in this recording.",
    "Are there multiple speakers in this audio? If so, how many?",
]

# Emotion Recognition Prompts
EMOTION_RECOGNITION_PROMPTS = [
    "What emotion is expressed in this speech?",
    "Identify the emotional tone of the speaker.",
    "What is the speaker's emotional state?",
    "Describe the emotion conveyed in this audio.",
]

# Sound Event Detection Prompts
SOUND_EVENT_DETECTION_PROMPTS = [
    "What events or actions are happening in this audio?",
    "Identify all the sound events in this audio clip.",
    "List the different sounds and events you can hear.",
    "What is happening in this audio recording?",
]

# Audio Comparison Prompts
AUDIO_COMPARISON_PROMPTS = [
    "Compare the two audio clips. What are the similarities and differences?",
    "How do these two audio recordings differ?",
    "What is similar between these audio clips?",
]

# Temporal Understanding Prompts
TEMPORAL_PROMPTS = [
    "Describe what happens in this audio over time.",
    "What is the sequence of events in this audio?",
    "Describe the temporal progression of sounds in this audio.",
]

# Audio Quality Assessment Prompts
QUALITY_ASSESSMENT_PROMPTS = [
    "Assess the audio quality of this recording.",
    "Is there any noise or distortion in this audio?",
    "Describe the recording quality and any issues.",
]


def get_speech_recognition_prompt(variation: int = 0) -> str:
    """Get a speech recognition prompt."""
    return SPEECH_RECOGNITION_PROMPTS[variation % len(SPEECH_RECOGNITION_PROMPTS)]


def get_audio_captioning_prompt(variation: int = 0) -> str:
    """Get an audio captioning prompt."""
    return AUDIO_CAPTIONING_PROMPTS[variation % len(AUDIO_CAPTIONING_PROMPTS)]


def get_sound_classification_prompt(options: list, variation: int = 0) -> str:
    """Get a sound classification prompt with options."""
    prompt_template = SOUND_CLASSIFICATION_PROMPTS[variation % len(SOUND_CLASSIFICATION_PROMPTS)]
    options_str = ", ".join(options)
    return prompt_template.format(options=options_str)


def get_audio_qa_prompt(question: str, variation: int = 0) -> str:
    """Get an audio QA prompt with a specific question."""
    prompt_template = AUDIO_QA_PROMPTS[variation % len(AUDIO_QA_PROMPTS)]
    return prompt_template.format(question=question)


def get_music_analysis_prompt(variation: int = 0) -> str:
    """Get a music analysis prompt."""
    return MUSIC_ANALYSIS_PROMPTS[variation % len(MUSIC_ANALYSIS_PROMPTS)]


def get_speaker_id_prompt(variation: int = 0) -> str:
    """Get a speaker identification prompt."""
    return SPEAKER_ID_PROMPTS[variation % len(SPEAKER_ID_PROMPTS)]


def get_emotion_recognition_prompt(variation: int = 0) -> str:
    """Get an emotion recognition prompt."""
    return EMOTION_RECOGNITION_PROMPTS[variation % len(EMOTION_RECOGNITION_PROMPTS)]


def get_sound_event_detection_prompt(variation: int = 0) -> str:
    """Get a sound event detection prompt."""
    return SOUND_EVENT_DETECTION_PROMPTS[variation % len(SOUND_EVENT_DETECTION_PROMPTS)]


# Example usage and task-specific prompt builders
def build_captioning_task(audio_description: str = "Audio clip") -> dict:
    """Build a complete captioning task prompt."""
    return {
        "pre_prompt": get_audio_captioning_prompt(),
        "audio_description": audio_description,
        "post_prompt": "Provide a detailed description of the sounds in the audio:",
    }


def build_transcription_task(audio_description: str = "Speech audio") -> dict:
    """Build a complete transcription task prompt."""
    return {
        "pre_prompt": get_speech_recognition_prompt(),
        "audio_description": audio_description,
        "post_prompt": "Transcription:",
    }


def build_classification_task(options: list, audio_description: str = "Audio") -> dict:
    """Build a complete classification task prompt."""
    return {
        "pre_prompt": get_sound_classification_prompt(options),
        "audio_description": audio_description,
        "post_prompt": "Answer:",
    }


def build_qa_task(question: str, audio_description: str = "Audio") -> dict:
    """Build a complete QA task prompt."""
    return {
        "pre_prompt": get_audio_qa_prompt(question),
        "audio_description": audio_description,
        "post_prompt": "Answer:",
    }
