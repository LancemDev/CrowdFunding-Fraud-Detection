    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lance/CODEWRLD/ICS4A/ML/crowdfundingproject/venv/lib/python3.13/site-packages/kagglehub/clients.py", line 182, in download_file
    kaggle_api_raise_for_status(response, resource_handle)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lance/CODEWRLD/ICS4A/ML/crowdfundingproject/venv/lib/python3.13/site-packages/kagglehub/exceptions.py", line 106, in kaggle_api_raise_for_status
    raise KaggleApiHTTPError(message, response=response) from e
kagglehub.exceptions.KaggleApiHTTPError: 404 Client Error.

Resource not found at URL: https://www.kaggle.com/datasets/kemical/kickstarter-projects/versions/7
The server reported the following issues: Dataset not found
Please make sure you specified the correct resource identifiers.