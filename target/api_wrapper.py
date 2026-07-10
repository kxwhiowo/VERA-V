from .base_target_lm import BaseTarget


class APIWrapper(BaseTarget):
    def __init__(self, model_name, num_tokens, temperature, top_k, client):
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.client = client

    def respond_to_query(self, query: str):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def batch_respond(self, queries):
        responses = []
        for q in queries:
            responses.append(self.respond_to_query(q))
        return responses

    # using the strong reject api  
    # for openai models, only need the model name
    # for huggingface models, need to specify the model and tokenizer
    def get_generation_pipeline(self): 
        return self.model_name
