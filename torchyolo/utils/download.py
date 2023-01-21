def attempt_download_from_hub(repo_id, hf_token=None):
    # https://github.com/fcakyon/yolov5-pip/blob/main/yolov5/utils/downloads.py
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils._errors import RepositoryNotFoundError
    from huggingface_hub.utils._validators import HFValidationError
    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type='model', token=hf_token)
        model_file = [f for f in repo_files if f.endswith('.pt')][0]
        file = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            repo_type='model',
            token=hf_token,
        )
        return file
    except (RepositoryNotFoundError, HFValidationError):
        return None