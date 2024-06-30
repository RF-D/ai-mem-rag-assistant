from langchain_community.document_loaders import GitHubIssuesLoader
from langchain.document_loaders import GithubFileLoader
import os

def load_github_issues(repo_owner, repo_name, access_token=None):
    """
    Load issues from a GitHub repository.

    :param repo_owner: The owner of the repository
    :param repo_name: The name of the repository
    :param access_token: GitHub personal access token (optional)
    :return: List of documents containing GitHub issues
    """
    if access_token is None:
        access_token = os.getenv("GITHUB_ACCESS_TOKEN")

    loader = GitHubIssuesLoader(
        repo=f"{repo_owner}/{repo_name}",
        access_token=access_token,
    )

    return loader.load()

def load_github_file(repo_owner, repo_name, file_path, branch="main", access_token=None):
    """
    Load a specific file from a GitHub repository.

    :param repo_owner: The owner of the repository
    :param repo_name: The name of the repository
    :param file_path: The path to the file within the repository
    :param branch: The branch to load the file from (default is "main")
    :param access_token: GitHub personal access token (optional)
    :return: List containing a single document with the file content
    """
    if access_token is None:
        access_token = os.getenv("GITHUB_ACCESS_TOKEN")

    loader = GithubFileLoader(
        repo=f"{repo_owner}/{repo_name}",
        file_path=file_path,
        branch=branch,
        access_token=access_token,
    )

    return loader.load()