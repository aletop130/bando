import requests

def ask_gpt_oss_120b_on_regolo(api_key, question):
    url = "https://api.regolo.ai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    if api_key:
        print("API key is valid")
    data = {
        "model": "Llama-3.3-70B-Instruct",
        "messages": [
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,
        "max_tokens": 150
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        resp_json = response.json()
        # Assuming the answer is in choices[0].message.content
        answer = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        return answer
    else:
        return f"Error: {response.status_code} - {response.text}"

# Usage example:
api_key = "sk-UeYc4J_9Mb9KTEUft8WzHw"
question = "What is Regolo?"
answer = ask_gpt_oss_120b_on_regolo(api_key, question)
print("Answer:", answer)
