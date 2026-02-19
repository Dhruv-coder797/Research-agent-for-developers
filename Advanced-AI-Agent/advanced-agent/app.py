import gradio as gr
from src.workflow import Workflow

workflow = Workflow()

def run_agent(query):
    result = workflow.run(query)
    return result.analysis

demo = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(
        label="Ask your research question",
        placeholder="Example: best frameworks for multi-agent systems"
    ),
    outputs=gr.Textbox(label="Agent Response"),
    title="Developer Tools Research Agent",
    description="AI agent that researches tools using Gemini + Firecrawl"
)

demo.queue().launch()
