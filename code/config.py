import os

def model_config(model: str):
    if model in ["deepseek-chat", "deepseek-reasoner"]:
        return {
            "model": model,
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        }
    elif model == "gpt-3.5-turbo":
        return {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        }
    else:
        return {
            "model": "gpt-4",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        }


def llm_config_list(seed, config_list):
    return {
        "functions": [
            {
                "name": "python",
                "description": "run cell in ipython and return the execution result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python cell to execute."
                        }
                    },
                    "required": ["cell"],
                },
            }
        ],
        "config_list": config_list,
        "timeout": 300,
        "cache_seed": seed,
        "temperature": 0,
    }