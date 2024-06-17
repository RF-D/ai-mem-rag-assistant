from langchain_community.document_loaders import YoutubeLoader


def youtube_chat(url):
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True
    )

    return loader.load()
