from diversity_metrics import lexical_diversity, syntactic_diversity, semantic_diversity, compute_self_bleu


# Compute diversity
# Example usage:
texts = [
    "Este es un ejemplo simple.",
    "Este ejemplo es simple y efectivo.",
    "Frase totalmente diferente y nada que ver."
]

lexical_score = lexical_diversity(texts)
print(f"Lexical Diversity Score: {lexical_score:.2f}")
syntactic_score = syntactic_diversity(texts)
print(f"Syntactic Diversity Score: {syntactic_score:.2f}")
semantic_score = semantic_diversity(texts)
print(f"Semantic Diversity Score: {semantic_score:.2f}")



# Example usage:
# texts = [
#     "The cat is on the mat.",
#     "There is a cat on the mat.",
#     "A cat sits on the mat.",
#     "The mat is under the cat."
# ]

# self_bleu_score = compute_self_bleu(texts)
# print(f"Self-BLEU score: {self_bleu_score}")