#!/usr/bin/env python3
# [MODULE DOC] Script that resolves the current default AWS credentials with boto3 and prints
# shell export lines for the access key id, secret key and session token, so a caller can
# evaluate the output to propagate credentials. Prints secrets to standard output.
import boto3  # type: ignore[import]


def main() -> None:
    # [FUNCTION DOC] Gets frozen credentials from the default boto3 session and prints three
    # export statements. Raises an attribute error if no credentials are found.
    creds_dict = boto3.Session().get_credentials().get_frozen_credentials()._asdict()
    print(f"export AWS_ACCESS_KEY_ID={creds_dict['access_key']}")
    print(f"export AWS_SECRET_ACCESS_KEY={creds_dict['secret_key']}")
    print(f"export AWS_SESSION_TOKEN={creds_dict['token']}")


if __name__ == "__main__":
    main()
