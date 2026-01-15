import random


def format_mcqa(sample, mcqa_type):
    choices = sample[f"{mcqa_type}_propositions"]
    random.shuffle(choices)
    letters = [chr(ord('A') + i) for i in range(len(choices))]
    answer_letter = letters[choices.index(sample[f"{mcqa_type}_answer"])]
    user_input = sample[f"{mcqa_type}_question"] + "".join([f"\n{letter}: {choice}" for choice, letter in zip(choices, letters)]) + "\nRéponse: "
    assistant_output = answer_letter
    return user_input, assistant_output