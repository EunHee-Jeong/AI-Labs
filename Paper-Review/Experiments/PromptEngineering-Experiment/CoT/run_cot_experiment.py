from transformers import pipeline
from datasets import load_dataset

GENERATION_CONFIG = {
    "max_new_tokens": 50,
    "temperature": 0.0, # CoT best practice
    # "top_k": 1,
    # "top_p": 0.0
}

def load_model():
    print("모델 로딩 중... (flan-t5-base)")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return generator

def generate(generator, prompt):
    return generator(prompt, **GENERATION_CONFIG)[0]["generated_text"]

questions = [
    "If there are 5 apples and you eat 2, how many are left?",
    "Sam went to the zoo and saw a lion, a tiger, and a bear. How many animals did he see?"
]

standard_prompts = [f"Q: {q}\nA:" for q in questions]
cot_prompts = [
    "Q: If there are 5 apples and you eat 2, how many are left?\nA: There are 5 apples. You eat 2. 5 - 2 = 3. So the answer is 3.",
    "Q: Sam went to the zoo and saw a lion, a tiger, and a bear. How many animals did he see?\nA: Sam saw a lion, a tiger, and a bear. That's 3 animals. So the answer is 3."
]

def run(generator):
    print("\n[Standard Prompts]")
    for i, prompt in enumerate(standard_prompts):
        output = generate(generator, prompt)
        print(f"[Standard-{i+1}]\n{output}\n")

    print("\n[CoT Prompts]")
    for i, prompt in enumerate(cot_prompts):
        output = generate(generator, prompt)
        print(f"[CoT-{i+1}]\n{output}\n")

def run_gsm8k(generator, sample_size=5):
    dataset = load_dataset("gsm8k", "main", split=f"test[:{sample_size}]")
    print(f"\n[GSM8K 실험 - 상위 {sample_size}개 문제 실행]")

    for i, item in enumerate(dataset):
        question = item["question"].strip()
        answer = item["answer"].strip().split("####")[-1].strip()

        standard_prompt = f"Q: {question}\nA:"
        standard_output = generate(generator, standard_prompt)

        cot_prompt = (f"Q: {question}\nA: Let's think step by step.")
        cot_output = generate(generator, cot_prompt)

        print(f"\n[{i + 1}] 질문: {question}")
        print(f"Standard Answer:\n{standard_output}")
        print(f"CoT Answer:\n{cot_output}")
        print(f"Gold Answer: {answer}")

if __name__ == "__main__":
    generator = load_model()
    run(generator)
    run_gsm8k(generator, sample_size=5)