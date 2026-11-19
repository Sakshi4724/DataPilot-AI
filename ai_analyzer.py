
import re
from ai_provider import generate_ai_text


def generate_ai_insights(summary=None, df=None):

    prompt = f"""
Analyze this dataset summary:

{summary}

Write a concise report using these five sections:

1. Dataset Overview
2. Statistical Observations
3. Data Quality
4. Recommended Visualizations
5. Conclusion

Rules:
- Maximum 150 words in total.
- Use simple English.
- Focus on important findings.
- Do not repeat every statistic.
- Never invent numerical values.
- CustomerID is an identifier, not a measurement.
- Unusual values are potential anomalies, not confirmed errors.
- Do not include planning or reasoning text.
- Return only the final report.
"""

    system_prompt = (
        "You are a professional data analyst. "
        "Return only a concise final data analysis report."
    )

    insights = generate_ai_text(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=1500
    )

    insights = re.sub(
        r"<think>.*?</think>",
        "",
        insights,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    if "</think>" in insights.lower():
        position = insights.lower().rfind("</think>")
        insights = insights[position + len("</think>"):].strip()

    if "<think>" in insights.lower():
        return "The AI response was incomplete. Please try again."

    if not insights:
        return "No AI report was returned. Please try again."

    return insights
