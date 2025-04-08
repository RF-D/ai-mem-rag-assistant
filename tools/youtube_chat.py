from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
import datetime
from pytube import exceptions as pytube_exceptions
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import logging
import re


def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    # Handle youtu.be URLs
    if "youtu.be" in url:
        video_id_match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
        if video_id_match:
            return video_id_match.group(1)
    
    # Handle youtube.com URLs
    video_id_match = re.search(r"(?:v=|/)([\w-]{11})", url)
    if video_id_match:
        return video_id_match.group(1)
    
    return None


def get_transcript(video_id):
    """Get the transcript for a YouTube video."""
    try:
        logging.info(f"Attempting to get transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript_list])
    except TranscriptsDisabled:
        logging.warning(f"Transcripts disabled for video ID: {video_id}")
        return "Transcript not available for this video."
    except Exception as e:
        logging.error(f"Error getting transcript for video ID {video_id}: {str(e)}")
        return "Error retrieving transcript."


def youtube_chat(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            logging.error(f"Could not extract video ID from URL: {url}")
            raise ValueError("Could not extract video ID from URL")

        logging.info(f"Extracted video ID: {video_id} from URL: {url}")
        
        # Get transcript first
        transcript_text = get_transcript(video_id)

        # Configure loader with additional options for resilience
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True, language=["en"], continue_on_failure=True
        )

        try:
            docs = loader.load()
            # Add transcript to the first document's content
            if docs and transcript_text:
                docs[0].page_content = (
                    f"{docs[0].page_content}\n\nTranscript:\n{transcript_text}"
                )
            return docs
        except pytube_exceptions.PytubeError as e:
            logging.error(f"PyTube error: {str(e)}")
            # Create a minimal document with just the URL and transcript if loading fails
            return [
                Document(
                    page_content=f"Video URL: {url}\n\nTranscript:\n{transcript_text}",
                    metadata={
                        "source": url,
                        "title": "Unknown",
                        "description": "Error loading video content",
                        "view_count": "Unknown",
                        "author": "Unknown",
                        "length": "Unknown",
                        "thumbnail_url": "Unknown",
                        "publish_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "publish_year": str(datetime.datetime.now().year),
                    },
                )
            ]

        for doc in docs:
            # Convert publish_date to a more usable format
            if "publish_date" in doc.metadata:
                try:
                    doc.metadata["publish_date"] = datetime.datetime.strptime(
                        doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    doc.metadata["publish_date"] = datetime.datetime.now().strftime(
                        "%Y-%m-%d"
                    )

            # Add publish year as a separate field
            if "publish_date" in doc.metadata:
                doc.metadata["publish_year"] = doc.metadata["publish_date"][:4]

            # Ensure all relevant metadata fields are present
            fields = [
                "title",
                "description",
                "view_count",
                "author",
                "length",
                "thumbnail_url",
            ]
            for field in fields:
                if field not in doc.metadata:
                    doc.metadata[field] = "Unknown"

        return docs

    except Exception as e:
        logging.error(f"Error processing YouTube URL {url}: {str(e)}")
        # Return a minimal document in case of any other errors
        return [
            Document(
                page_content=f"Error processing video: {url}",
                metadata={
                    "source": url,
                    "title": "Error",
                    "description": f"Error: {str(e)}",
                    "view_count": "Unknown",
                    "author": "Unknown",
                    "length": "Unknown",
                    "thumbnail_url": "Unknown",
                    "publish_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "publish_year": str(datetime.datetime.now().year),
                },
            )
        ]
