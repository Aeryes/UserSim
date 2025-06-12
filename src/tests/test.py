import boto3
import json

def run_bedrock_test():
    # If using a named local profile with Bedrock permissions
    session = boto3.Session(profile_name="bedrock-local", region_name="us-east-1")
    client = session.client("bedrock-runtime")

    model_id = "anthropic.claude-v2"

    body = {
        "prompt": "\n\nHuman: Say hello to me like a robot.\n\nAssistant:",
        "max_tokens_to_sample": 100,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 0.95,
        "stop_sequences": ["\n\nHuman:"]
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response['body'].read())
    print("🧠 Bedrock LLM Response:")
    print(result['completion'])

if __name__ == "__main__":
    run_bedrock_test()
