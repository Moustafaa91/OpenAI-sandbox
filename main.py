import openai
import os

from dotenv import load_dotenv, find_dotenv
from reviews import all_reviews;
_ = load_dotenv(find_dotenv())

client = openai.OpenAI()
openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-5.4-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


for i in range(len(all_reviews)):
    prompt = f"""
        Your task is to generate a short summary of a product 
        review from an ecommerce site. and give it a score from 1 to 10, where 1 is the worst and 10 is the best. \

        Summarize the review below, delimited by triple
        backticks in at most 20 words.
        Structured output format is JSON:
        {{"summary": "<summary>", "score": <score>}}

        Review: ```{all_reviews[i]}```
        """

    try:
        response = get_completion(prompt)
        print(response)
    except Exception as e:
        print(f"OpenAI API error: {e}")

