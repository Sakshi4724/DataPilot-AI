
import re
import ollama

MODEL_NAME = "qwen3:4b"


def generate_ai_insights(summary=None, df=None):

    prompt = f"""
/no_think

You are a professional data analyst.

Analyze this dataset summary:

{summary}

Write a concise report using these five sections:

1. Dataset Overview
2. Statistical Observations
3. Data Quality
4. Recommended Visualizations
5. Conclusion

Rules:
- Write a maximum of 150 words in total.
- Use simple English.
- Focus on important findings.
- Do not repeat every statistic.
- Never invent numerical values.
- CustomerID is an identifier, not a measurement.
- Unusual values are potential anomalies, not confirmed errors.
- Do not include planning or reasoning text.
- Return only the final report.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "Return a concise final data analysis report only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 1500
        },
        think=False
    )

    insights = response["message"]["content"].strip()

    # Remove complete reasoning blocks, if present.
    insights = re.sub(
        r"<think>.*?</think>",
        "",
        insights,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Remove reasoning preceding a closing tag.
    if "</think>" in insights.lower():
        position = insights.lower().rfind("</think>")
        insights = insights[position + len("</think>"):].strip()

    # Avoid displaying unfinished reasoning.
    if "<think>" in insights.lower():
        return "The AI response was incomplete. Please try again."

    if not insights:
        return "No AI report was returned. Please try again."

    return insights
