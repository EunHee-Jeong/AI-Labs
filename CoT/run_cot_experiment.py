from transformers import pipeline

# 문제 정의
questions = [
    "If there are 5 apples and you eat 2, how many are left?",
    "Sam went to the zoo and saw a lion, a tiger, and a bear. How many animals did he see?"
]

standard_prompts = [f"Q: {q}\nA:" for q in questions]
cot_prompts = [
    "Q: If there are 5 apples and you eat 2, how many are left?\nA: There are 5 apples. You eat 2. 5 - 2 = 3. So the answer is 3.",
    "Q: Sam went to the zoo and saw a lion, a tiger, and a bear. How many animals did he see?\nA: Sam saw a lion, a tiger, and a bear. That's 3 animals. So the answer is 3."
]

# 모델 불러오기
def load_model():
    print("모델 로딩 중... (flan-t5-base)")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return generator

# 실행 함수
def run():
    generator = load_model()

    print("\n[Standard Prompts]")
    for i, prompt in enumerate(standard_prompts):
        output = generator(prompt, max_new_tokens=50)[0]["generated_text"]
        print(f"[Standard-{i+1}]\n{output}\n")

    print("\n[CoT Prompts]")
    for i, prompt in enumerate(cot_prompts):
        output = generator(prompt, max_new_tokens=50)[0]["generated_text"]
        print(f"[CoT-{i+1}]\n{output}\n")

# main 진입점
if __name__ == "__main__":
    run()
