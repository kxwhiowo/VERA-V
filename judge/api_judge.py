from .base_judge import BaseJudge


class APIJudge(BaseJudge): 
    def __init__(self, model_name, num_tokens, temperature, top_k, client, max_val):
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.client = client
        self.max_val = max_val

    def get_response(self, system, user): 
        """
        Generic response getter for judge models when we use OpenAI API.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return response.choices[0].message.content 