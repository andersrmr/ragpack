"""Make one real request to verify the provider connection."""

import json
import sys

from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from tinyagent import LLM


def main() -> int:
    llm = None
    try:
        llm = LLM()
        response = llm.generate([
            {"role": "user", "content": "Say hello in one short sentence."}
        ])
        print(response.content)
        print(json.dumps(response.metadata, indent=2))
        return 0
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
    except AuthenticationError:
        print("Authentication failed. Check OPENAI_API_KEY.", file=sys.stderr)
    except RateLimitError:
        print("Request limited. Check your API quota, billing, and rate limits.", file=sys.stderr)
    except APIConnectionError:
        print("Could not connect to OpenAI. Check your network connection.", file=sys.stderr)
    except APIStatusError as error:
        print(
            f"OpenAI returned HTTP {error.status_code}. Check that TINYAGENT_MODEL "
            "supports Chat Completions and is available to your account.",
            file=sys.stderr,
        )
    except RuntimeError as error:
        print(f"Response error: {error}", file=sys.stderr)
    finally:
        if llm is not None:
            llm.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
