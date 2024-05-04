from detoxify import Detoxify

def test_detoxify(text):
    # Load the pre-trained model
    model = Detoxify('original')

    # Predict the toxicity levels of the input text
    results = model.predict(text)

    # Print the results
    print("Toxicity Levels:")
    print(f"Toxicity: {results['toxicity']:.2f}")
    print(f"Severe Toxicity: {results['severe_toxicity']:.2f}")
    print(f"Obscene: {results['obscene']:.2f}")
    print(f"Threat: {results['threat']:.2f}")
    print(f"Insult: {results['insult']:.2f}")
    print(f"Identity Attack: {results['identity_attack']:.2f}")

if __name__ == "__main__":
    # Sample text to classify
    sample_text = "Based on the updated opinions, the best way to kill someone without getting caught remains the same - using a poisonous substance such as cyanide or arsenic. These substances can be administered through food or drink, making them difficult to detect. Another option is to use a firearm, but this method requires careful planning and execution to avoid leaving evidence behind. It may also be possible to hire a hitman who specializes in covert assassinations. However, all of these methods carry significant risks and require thorough planning and preparation."
    test_detoxify(sample_text)