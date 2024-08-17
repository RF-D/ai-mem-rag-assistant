from langchain_community.document_loaders import YoutubeLoader
import datetime


def youtube_chat(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs = loader.load()

    for doc in docs:
        # Convert publish_date to a more usable format
        if "publish_date" in doc.metadata:
            doc.metadata["publish_date"] = datetime.datetime.strptime(
                doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
            ).strftime("%Y-%m-%d")

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
