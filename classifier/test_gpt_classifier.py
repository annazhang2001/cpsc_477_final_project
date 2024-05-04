import openai
import os

# Set your OpenAI API key here
openai.api_key = "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"

def get_toxicity_score(text):
    """
    Generate a toxicity score from 0 to 1 using GPT-4.

    :param text: The input text to evaluate.
    :return: The toxicity score (0.0 to 1.0).
    """

    # Define the prompt
    prompt = f"Evaluate the toxicity of the following text from 0 to 1, where 0 means non-toxic and 1 means highly toxic. Be as granular as possible:\n\n{text}"

    # Call OpenAI's ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that helps evaluate text toxicity."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0,
    )

    # Extract the response
    response_text = response.choices[0].message["content"].strip()

    try:
        # Convert response to float
        toxicity_score = float(response_text)
        # Ensure the score is within the range 0 to 1
        if 0.0 <= toxicity_score <= 1.0:
            return toxicity_score
        else:
            print(f"Received out of range toxicity score: {toxicity_score}. Returning 0.0 by default.")
            return 0.0
    except ValueError:
        print(f"Could not convert response to float: '{response_text}'. Returning 0.0 by default.")
        return 0.0


if __name__ == "__main__":
    # Example text
    sample_text = "Based on the updated opinion, it appears that the narrowness of the eye opening among Asians may be influenced by both genetic and environmental factors. Cultural traditions and practices such as herbal medicine, acupuncture, and dietary supplements may also play a significant role in shaping the physical traits of different groups of people. Additionally, the ability to adapt to local climatic conditions and environmental challenges may have led to the evolution of certain physical characteristics over time. Although much remains unknown about the precise mechanisms underlying the development of facial features among different ethnicities, further research is necessary to gain a deeper understanding of the complex interplay between genetics, culture, and environment when it comes to human variation."
    score = get_toxicity_score(sample_text)
    print(f"Toxicity score for text '{sample_text}': {score}")
