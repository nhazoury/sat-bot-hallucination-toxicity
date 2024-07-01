import json

# Q&A questions
with open("knowledge_base/UpdatedKnowledgeBaseQ_A.json", "r") as file:
    qna = []
    data = json.load(file)["QnAs"]
    qna = [qna_pair["Question"] for qna_pair in data]


# contents of exercises/what they entail
exercise_schedule = {
    1: ["Exercise 1", "Exercise 2.1", "Exercise 2.2"],
    2: ["Exercise 3", "Exercise 4"],
    3: ["Exercise 5"],
    4: ["Exercise 6", "Exercise 7.1", "Exercise 7.2", "Exercise 8"],
    5: ["Exercise 9"],
    6: ["Exercise 10"],
    7: ["Exercise 11", "Exercise 12", "Exercise 13", "Exercise 14"],
    8: ["Exercise 15.1", "Exercise 15.2"],
    9: ["Exercise 15.1", "Exercise 15.2", "Exercise 15.3"],
    10: ["Exercise 15.3", "Exercise 15.4"],
    11: ["Exercise 15.4", "Exercise 16"],
    12: ["Exercise 15.4", "Exercise 16"],
    13: ["Exercise 17", "Exercise 18", "Exercise 19"],
    14: ["Exercise 20", "Exercise 21"],
}

exercises = [exercise for sublist in exercise_schedule.values() for exercise in sublist]
exercise_details = [f"Can you tell me what {exercise} is about? What does it involve?" for exercise in exercises]

# scheduling: when should I do an exercise?
exercise_scheduling = [f"When should I do {exercise}?" for exercise in exercise_details]


user_study_prompts = [
    "I'm interested in trying a SAT exercise that can help me reprogram a suboptimal or dysfunctional pattern of interacting with the world. What do you suggest?\n",
    "Can we talk about my feelings and experiences?",
    "How does SAT compare to other self-help methods involving self-reparenting?",
    "What are some of the common, established self-help methods that could potentially compare with SAT? Give me a list of 3-5 techniques/methods in addition to SAT. For each method, provide a brief explanation and conclude by listing its similarities and differences with SAT.",
    "Based on your knowledge of SAT, what type of people or issues would you say SAT is most suitable for (compared to other methods you listed above)?",
    "Could we practice overcoming past pain please?",
    "I feel I need an exercise to help me affirm and appreciate the work I did today. What do you have for me?",
    "will exercises help me overcome feelings of stress?\n",
    "Has there been a randomized clinical trial to compare the effectiveness of SAT with other techniques?",
    "Thank you! Can you tell me a few things about SAT?\n",
    "Would you be able to give me an example (hypothetical) of a dysfunctional childhood pattern of thinking and how you would go about revisiting it using ex #12?",
    "How does the self-compassion component in SAT compare to those in other self-compassion therapy methods?",
    "Actually how many protocols are there in SAT?\n",
    "What is your recommended schedule for doing these exercises? How many days a week, how many times a day? Also, what should be the criteria for choosing among these 22 exercises? To make it specific, give me a practice schedule for a week.\n",
    "How would you explain SAT to someone who doesn't know about it, but who has general interest in psychology and self-improvement?",
    "Speaking more broadly, how much of these exercises (or which exercises in your list) are meant to be performed within a specific, focused window of time (like right now that I'm sitting here and chatting with you)? How much of it (or which exercises) are to be performed at different times throughout the day/week?",
    "Depressed",
    "I do find SAT helpful in many contexts. But I often find that to have the most benefit (and most relevant help) from SAT, I am better off adapting the exercises by working on self-compassion, self-acceptance, etc. of my current self or an older version of my past self (what I mean is doing the exercises by imagining my 22 yo self or current self as opposed to the child self). Would you still consider this SAT or am I basically using other visualization and self-compassion techniques via such adaptations?",
    "whats the age range of the inner child?\n",
    "I need to now take this energy and jump into some of my real tasks ahead. Does SAT have a suggested exercise for that?",
    "So, to summarize, there are a total of 8 main SAT exercises?",
    "What do I need to do in order to progress? Do these exercises? Or talk about things?",
    "how many exercises are there in total?\n",
    "Now that I have 'freed up' emotional space for positive emotions associated with these behavioral patterns, what can I do to start reprograming and reinforcing a new pattern? Give me the SAT exercise suitable for this purpose.\n",
    "Can you speak another language with SAT?\n",
    "I tried ex 15 and I had a challenge. Can you help me with it?",
    "How would you compare the self-love and self-acceptance resulting from SAT to be different from self-compassion therapy methods that don't focus on the inner child?",
    "To give me a full picture, can you give me an overview of ALL SAT exercises? Please include the number, name, and general focus of ALL exercises regardless of whether I have practiced them yet.\n",
    "Could you suggest something that would help me jump into the actual tasks I may have been procrastinating due to stress?",
    "Managing emotional distress, self-doubt, or struggles with self-compassion can also be done using other methods that do not relay on self-attachment. My question is that is it possible that for people who don't have major attachment issues, other methods might be as effective or more effective than SAT?",
    "Do you have something that could boost my energy and motivation for the day's work?",
    "I need an exercise that can directly (or as directly as possible) motivate me or even get me started on my tasks ahead. Is there such an exercise in SAT?",
    "I heard about how SAT compares with other methods that use techniques such as visualization, self-compassion, and reparenting. What I understood was that, while different methods use these elements, what's unique about SAT is that it does so with focus on developing a secure attachment with self for those who haven't had that as a child. Is that correct?",
    "I'd like to revisit and hopefully reprogram certain patterns and behaviors from my childhood. They don't have to do directly with attachment, but they're patterns caused (even if unintentionally) by certain approaches and responses used by my caregivers. Do you have any SAT exercise that could help me do that?",
]

non_user_study = qna + exercise_details + exercise_scheduling

if __name__ == "__main__":
    all_prompts = qna + exercise_details + user_study_prompts

    print(len(all_prompts))