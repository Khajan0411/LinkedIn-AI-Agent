import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

# News Generation Agent
web_search_agent = Agent(
    name="AI News LinkedIn Curator",
    role="Create a professional LinkedIn post about a single AI news topic by gathering information from multiple sources and rewriting it.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=[
        "Search for the latest news on a specific AI topic (e.g., a new AI model release).",
        "Gather information from at least 3 different sources.",
        "Rewrite the information into a single, cohesive, and engaging LinkedIn post.",
        "Write in a professional tone suitable for LinkedIn.",
        "Include key details such as:",
        "  - What the development is",
        "  - Why it is significant",
        "  - How it impacts the industry or society",
        "  - Any relevant statistics or quotes",
        "Use bullet points or short paragraphs for readability.",
        "Include relevant hashtags (e.g., #AI, #ArtificialIntelligence, #TechNews).",
        "End with a question or call-to-action to encourage engagement.",
        "Ensure the post is under 3000 characters.",
    ],
    show_tools_calls=True,
    markdown=True
)

# News Relevance Agent
news_relevance_agent = Agent(
    name="News Relevance Validator",
    role="Critically evaluate the rewritten AI news content for LinkedIn posting suitability.",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "Carefully assess the rewritten AI news content.",
        "Ensure the content is focused on a single topic and is cohesive.",
        """Check for:
            - Professionalism
            - Accuracy of information
            - Absence of controversial content
            - Engagement potential""",
        """Provide a structured evaluation with:
            - Suitability score (0-10)
            - Posting recommendation (Yes/No)
            - Specific reasons for evaluation""",
        "If not suitable, explain specific reasons and suggest modifications.",
        "Respond with 'No' in the posting recommendation if content is not suitable.",
    ],
    show_tools_calls=True,
    markdown=True
)

def main():
    # Define the specific AI topic to search for
    topic = "latest AI model release"  # You can change this to any specific topic

    # Generate AI news content
    news_response = web_search_agent.run(
        f"Search for the latest news on '{topic}' from multiple sources and rewrite it into a professional LinkedIn post.", 
        stream=False
    )
    
    # Validate the generated news content
    validation_response = news_relevance_agent.run(
        f"Evaluate the following AI content for LinkedIn posting suitability:\n\n{news_response.content}", 
        stream=False
    )
    
    # Check if validation recommends not posting
    news_content = news_response.content
    if "<function=duckduckgo_news" in validation_response.content:
        news_content = ""
    else:
        news_content = news_response.content
    
    return {
        "news_content": news_content,
        "validation": validation_response.content
    }

if __name__ == '__main__':
    result = main()
    print("Generated News:")
    print(result['news_content'])
    print("\nValidation Result:")
    print(result['validation'])