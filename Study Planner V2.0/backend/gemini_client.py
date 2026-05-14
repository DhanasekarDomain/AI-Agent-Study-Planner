import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

# -------------------------------
# DuckDuckGo Search Helper
# -------------------------------
def perform_web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo search and return a list of results."""
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                if not isinstance(result, dict):
                    continue
                title = result.get('title') or ''
                href = result.get('href') or ''
                body = result.get('body') or ''
                if title and href:
                    results.append({
                        'title': title,
                        'href': href,
                        'body': body,
                    })
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

# -------------------------------
# Base Gemini Client
# -------------------------------
class GeminiClient:
    def __init__(self):
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            # Updated to 2.5-flash as requested. 
            # Note: Ensure your SDK supports this specific string.
            self.model = genai.GenerativeModel(
                model_name='gemini-2.5-flash', 
                system_instruction=(
                    "You are a professional AI Study Planner. "
                    "Output must be in clean Markdown. "
                    "Use ## for headings, **bold** for key terms, and standard bullet points (-) "
                    "or numbered lists (1.) for steps. "
                    "Do not add extra symbols like arrows or custom emojis unless requested."
                )
            )
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            self.chat = None

    def generate_response(self, user_input: str) -> str:
        """Generate an AI response with optional web search when prefixed."""
        if not self.chat:
            return "⚠️ AI service is not configured correctly."

        try:
            text = user_input or ""
            lower = text.strip().lower()

            search_query = None
            if lower.startswith("search:"):
                search_query = text.split(":", 1)[1].strip()
            elif lower.startswith("/search "):
                search_query = text.split(" ", 1)[1].strip()

            if search_query:
                web_results = perform_web_search(search_query, max_results=6)
                if not web_results:
                    return "⚠️ Web search failed to retrieve results."

                refs_lines = []
                for idx, item in enumerate(web_results, start=1):
                    # Building a clean context block for the AI
                    refs_lines.append(f"Source {idx}: {item['title']}\nURL: {item['href']}\nSnippet: {item['body']}")
                
                refs_block = "\n\n".join(refs_lines)
                composed = (
                    f"Answer the user query: '{search_query}' based on these web results:\n\n{refs_block}"
                )
                response = self.chat.send_message(composed)
                return response.text # Raw Markdown for the frontend

            response = self.chat.send_message(text)
            return response.text # Raw Markdown for the frontend

        except Exception as e:
            print(f"Error generating response: {e}")
            return "⚠️ Encountered an error processing your request."

# -------------------------------
# Specialized Agents
# -------------------------------
class PlannerAgent(GeminiClient):
    def plan_study(self, subject: str) -> str:
        return self.generate_response(f"Create a high-level study plan for {subject}.")

class SchedulerAgent(GeminiClient):
    def schedule(self, subject: str, hours: int = 10) -> str:
        return self.generate_response(f"Create a {hours}-hour weekly schedule for studying {subject}.")

class ReviewerAgent(GeminiClient):
    def review(self, topic: str) -> str:
        return self.generate_response(f"Provide a summary and a 5-question quiz for {topic}.")

class MotivationAgent(GeminiClient):
    def motivate(self) -> str:
        return self.generate_response("Give me motivational advice for a student.")

class ResourceAgent(GeminiClient):
    def find_resources(self, topic: str) -> str:
        return self.generate_response(f"search: best educational resources for {topic}")

# -------------------------------
# Pipeline Manager
# -------------------------------
class StudyPipeline:
    def __init__(self):
        self.planner = PlannerAgent()
        self.scheduler = SchedulerAgent()
        self.reviewer = ReviewerAgent()
        self.motivator = MotivationAgent()
        self.resources = ResourceAgent()

    def run_pipeline(self, subject: str, hours: int, review_topic: str) -> str:
        """Runs all agents and combines them with Markdown separators."""
        results = [
            "## 📘 Study Plan", self.planner.plan_study(subject),
            "---",
            "## 🗓️ Schedule", self.scheduler.schedule(subject, hours),
            "---",
            "## 📝 Review & Quiz", self.reviewer.review(review_topic),
            "---",
            "## 💡 Motivation", self.motivator.motivate(),
            "---",
            "## 🔗 Resources", self.resources.find_resources(subject)
        ]
        return "\n\n".join(results)

if __name__ == "__main__":
    pipeline = StudyPipeline()
    print(pipeline.run_pipeline("Aerodynamics", 10, "Bernoulli's Principle"))