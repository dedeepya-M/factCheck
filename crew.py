from crewai import  Crew, Process
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from agents import MisinformationDetectionAgent
from tasks import MisinformationDetectionTasks
from crewai import  Crew, Process



load_dotenv()


class AgentCrew:
    
    def run(self,prompt,path):
        
        agents = MisinformationDetectionAgent()
        tasks = MisinformationDetectionTasks()
        
        text_agent = agents.text_agent()
        image_agent = agents.image_agent()
        manager_agent = agents.manager_agent()
        
        text_task = tasks.text_task()
        image_task = tasks.image_task()
        
        
        crew = Crew(
        agents=[image_agent,text_agent],
        tasks=[image_task,text_task],
        manager_agent=manager_agent,
        process=Process.hierarchical,
        full_output=True,
        verbose=True,
        )
        if(prompt and path):
            results = crew.kickoff(inputs={'message':prompt,'image_path': None})
        else:        
            results = crew.kickoff(inputs={'message':prompt,'image_path': path})
        
        return results['final_output']
    
# prompt = """s
# cocaine kills corona virus
# """
prompt = None
path = 'fake\cocaine.png'
# path = None  

model = AgentCrew()

response = model.run(prompt,path)

print(response)    