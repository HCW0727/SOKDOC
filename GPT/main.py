import openai

openai.api_key = "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"
messages= [{"role": "system", "content": "You are a program that determines the possibility of a user having a stroke. Of course, the information you give me is not professional medical information, so I shouldn't believe it, but I'll just refer to it. Think of the user as an elderly person, ask the user one question at a time about the symptoms of stroke, inferring whether it is a stroke from the user's answer, and then determine the probability of the user's stroke. Please answer or ask a question in one sentence in korean"}]
#messages.append({"role" : "system", "content": "If you can't answer with just the information you're given, you're an analyst who tells you what you need additional information"})
while True:
    user_content = input("user : ")
    messages.append({"role": "user", "content": f"{user_content}"})

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    assistant_content = completion.choices[0].message["content"].strip()

    messages.append({"role": "assistant", "content": f"{assistant_content}"})

    print(f"GPT : {assistant_content}")