# Quick Start Guide: Launch Your AI Research Paper Agent

This guide will help you get the AI research paper agent up and running in minutes.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- Internet connection

## Step-by-Step Instructions

### 1. Set Up Your Environment

First, create a virtual environment and activate it:

```bash
# Create virtual environment
python -m venv ai-research-agent

# Activate it (choose based on your OS)
# On macOS/Linux:
source ai-research-agent/bin/activate

# On Windows:
ai-research-agent\Scripts\activate
```

### 2. Install Dependencies

Install the required packages:

```bash
pip install openai arxiv sentence-transformers requests
```

### 3. Set Your API Key

Set your OpenAI API key as an environment variable:

```bash
# On macOS/Linux:
export OPENAI_API_KEY='your-api-key-here'

# On Windows (Command Prompt):
set OPENAI_API_KEY=your-api-key-here

# On Windows (PowerShell):
$env:OPENAI_API_KEY='your-api-key-here'
```

### 4. Create the Project Structure

Create the necessary folders and files:

```bash
# Create directory structure
mkdir -p src/research src/writing output
touch src/__init__.py src/research/__init__.py src/writing/__init__.py
```

### 5. Add the Code Files

Copy the code from the guide into these files:

**src/research/literature_search.py** - Copy the `LiteratureSearcher` class from the guide

**src/research/summarization.py** - Copy the `PaperSummarizer` class

**src/writing/section_generator.py** - Copy the `SectionGenerator` class

**src/writing/citation_manager.py** - Copy the `CitationManager` class

**src/orchestrator.py** - Copy the `ResearchPaperAgent` class

**main.py** - Copy the main entry point code

### 6. Launch the Agent

Run the agent:

```bash
python main.py
```

The agent will:
1. Search for relevant papers on your topic
2. Analyze and summarize them
3. Generate a complete research paper
4. Save it to `output/research_paper.md`

## Quick Example

If you want to test quickly without creating all files, here's a minimal working example:

```python
# test_agent.py
import os
from openai import OpenAI

# Set your API key
os.environ['OPENAI_API_KEY'] = 'your-key-here'

client = OpenAI()

# Generate a simple abstract
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": """Write an academic abstract for a research paper on:
        Topic: Impact of AI on Education
        Length: 200 words
        Include: Background, Methods, Results, Conclusion"""
    }]
)

print(response.choices[0].message.content)
```

Run it:
```bash
python test_agent.py
```

## Customizing Your Paper

Edit `main.py` to change the research topic:

```python
# Define your research parameters
topic = "Your Research Topic Here"
research_question = "Your specific research question?"

requirements = {
    'gap': 'What gap does this address?',
    'contribution': 'What does your research contribute?',
    'methodology': 'What methods will you use?',
    'findings': 'What are the expected findings?'
}
```

## Troubleshooting

### "No module named 'openai'"
Install the package: `pip install openai`

### "Please set OPENAI_API_KEY environment variable"
Make sure you've set your API key correctly (Step 3)

### "Rate limit exceeded"
You're making too many API calls. Wait a few minutes or upgrade your OpenAI plan.

### Import errors
Make sure all `__init__.py` files are created in the directories

## Cost Estimate

Generating one research paper typically costs:
- Using GPT-4: $1-3 per paper
- Using GPT-3.5-Turbo: $0.10-0.30 per paper

## Next Steps

1. Read the full [AI-RESEARCH-PAPER-AGENT-GUIDE.md](./AI-RESEARCH-PAPER-AGENT-GUIDE.md) for detailed explanations
2. Customize the prompts in `section_generator.py`
3. Add more data sources in `literature_search.py`
4. Implement quality checks in a review module
5. Deploy as a web service using Flask (see guide)

## Need Help?

- Full documentation: [AI-RESEARCH-PAPER-AGENT-GUIDE.md](./AI-RESEARCH-PAPER-AGENT-GUIDE.md)
- OpenAI API docs: https://platform.openai.com/docs
- Python setup: https://www.python.org/downloads/

Happy researching! 🚀
