import os
import re
import time
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import chromadb
import uuid
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
        self.email_templates = {
            "formal": "You are Mohan, a business development executive at AtliQ. Write a formal and professional cold email to the client regarding the job mentioned above describing AtliQ's capability in fulfilling their needs.",
            "conversational": "You are Mohan, a business development executive at AtliQ. Write a friendly and conversational cold email to the client that shows personality while highlighting AtliQ's capabilities for the job above.",
            "problem-solution": "You are Mohan, a business development executive at AtliQ. Write a cold email that identifies specific problems the client might be facing based on the job description, and position AtliQ's solutions as the answer."
        }
        
        self.company_research_cache = {}

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, `description`, and `company_name` (if available).
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def extract_company_info(self, url, company_name=None):
        """Extract company information from the URL or by searching for the company name"""
        if url in self.company_research_cache:
            return self.company_research_cache[url]
            
        base_url = '/'.join(url.split('/')[:3])
        about_url = f"{base_url}/about"
        
        try:
            # Try to find About page
            response = requests.get(about_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                about_text = soup.get_text()
                about_text = ' '.join(about_text.split())[:5000]  # Truncate to prevent token overload
            else:
                about_text = "No company information found."
            
            # If company name is available, try to get additional info
            company_info = f"Company: {company_name if company_name else 'Unknown'}\nAbout: {about_text}"
            
            # Extract company info using LLM
            prompt = PromptTemplate.from_template(
                """
                ### COMPANY INFO:
                {company_info}
                
                ### INSTRUCTION:
                Extract key information about this company that would be useful for a cold email. Focus on:
                1. Company values
                2. Recent initiatives or projects 
                3. Company pain points based on industry
                4. Company size and scale
                
                Format the response as JSON with these keys: values, initiatives, pain_points, size.
                Only return the valid JSON.
                """
            )
            
            chain = prompt | self.llm
            res = chain.invoke({"company_info": company_info})
            
            try:
                json_parser = JsonOutputParser()
                extracted_info = json_parser.parse(res.content)
                # Cache the result
                self.company_research_cache[url] = extracted_info
                return extracted_info
            except:
                return {"values": [], "initiatives": [], "pain_points": [], "size": "Unknown"}
                
        except Exception as e:
            print(f"Error extracting company info: {e}")
            return {"values": [], "initiatives": [], "pain_points": [], "size": "Unknown"}

    def write_mail(self, job, links, template_style="formal", personalization=None):
        # Get company info if available
        company_name = job.get('company_name', None)
        company_info = ""
        
        if personalization and personalization.get('include_company_research', False):
            company_research = self.extract_company_info(personalization.get('company_url', ''), company_name)
            if company_research:
                company_info = f"""
                Company Values: {', '.join(company_research.get('values', [])[:3])}
                Recent Initiatives: {', '.join(company_research.get('initiatives', [])[:2])}
                Potential Pain Points: {', '.join(company_research.get('pain_points', [])[:2])}
                Company Size: {company_research.get('size', 'Unknown')}
                """
        
        # Create the template instruction based on style
        template_instruction = self.email_templates.get(template_style, self.email_templates["formal"])
        
        # Personalization options
        recipient_name = personalization.get('recipient_name', '') if personalization else ''
        add_call_to_action = personalization.get('add_call_to_action', False) if personalization else False
        mention_competitors = personalization.get('mention_competitors', False) if personalization else False
        
        prompt_email = PromptTemplate.from_template(
            f"""
            ### JOB DESCRIPTION:
            {{job_description}}

            ### COMPANY RESEARCH:
            {company_info}

            ### INSTRUCTION:
            {template_instruction}
            
            AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency.
            
            {"Address the email to " + recipient_name if recipient_name else "Address the email appropriately."}
            
            {"Include a clear call to action at the end of the email, such as suggesting a meeting time or requesting a response." if add_call_to_action else ""}
            
            {"Briefly mention how AtliQ's solutions compare favorably to competitors in the industry." if mention_competitors else ""}
            
            Add the most relevant ones from the following links to showcase Atliq's portfolio: {{link_list}}
            
            Remember you are Mohan, BDE at AtliQ.
            Format the email properly with subject line, greeting, body, and signature.
            
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

    def generate_follow_up(self, original_email, days_passed=7):
        prompt_followup = PromptTemplate.from_template(
            """
            ### ORIGINAL EMAIL:
            {original_email}
            
            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ. The recipient hasn't responded to your email sent {days} days ago.
            Write a brief follow-up email that:
            1. References the original email
            2. Adds a new piece of value or information
            3. Gently asks for a response
            4. Maintains a professional but not pushy tone
            
            Format the email properly with subject line, greeting, body, and signature.
            The subject should indicate this is a follow-up.
            
            Do not provide a preamble.
            ### FOLLOW-UP EMAIL (NO PREAMBLE):
            """
        )
        chain_followup = prompt_followup | self.llm
        res = chain_followup.invoke({"original_email": original_email, "days": days_passed})
        return res.content


class Portfolio:
    def __init__(self, file_path="app/resource/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills, n_results=2):
        return self.collection.query(query_texts=skills, n_results=n_results).get('metadatas', [])
    
    def add_portfolio_item(self, techstack, link):
        """Add a new portfolio item"""
        # Add to dataframe
        new_row = pd.DataFrame({"Techstack": [techstack], "Links": [link]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self.data.to_csv(self.file_path, index=False)
        
        # Add to vector database
        self.collection.add(
            documents=[techstack],
            metadatas=[{"links": link}],
            ids=[str(uuid.uuid4())]
        )
        return True


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def save_email_history(email, job_info, url, template_style):
    """Save generated email to history"""
    history_file = "email_history.json"
    
    # Create entry
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "job_title": job_info.get("role", "Unknown Role"),
        "company": job_info.get("company_name", "Unknown Company"),
        "url": url,
        "template_style": template_style,
        "email": email
    }
    
    # Load existing history or create new
    try:
        with open(history_file, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    
    # Add new entry and save
    history.append(entry)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def load_email_history():
    """Load email history"""
    history_file = "email_history.json"
    try:
        with open(history_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def create_streamlit_app():
    st.set_page_config(layout="wide", page_title="Cold Email Generator Pro", page_icon="📧")
    
    # Initialize objects
    chain = Chain()
    portfolio = Portfolio()
    portfolio.load_portfolio()
    
    # Title and sidebar
    st.title("📧 Cold Email Generator Pro")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Generate Email", "Portfolio Manager", "Email History", "Settings"])
    
    # Tab 1: Generate Email
    with tab1:
        st.header("Generate New Cold Email")
        
        # Input form
        col1, col2 = st.columns([3, 1])
        
        with col1:
            url_input = st.text_input("Enter Job URL:", value="https://jobs.nike.com/job/R-33460")
            
            # More advanced options in an expander
            with st.expander("Personalization Options"):
                email_style = st.selectbox(
                    "Email Style",
                    options=["formal", "conversational", "problem-solution"],
                    index=0
                )
                
                recipient_name = st.text_input("Recipient Name (if known):", value="")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    include_company_research = st.checkbox("Include Company Research", value=True)
                with col_b:
                    add_call_to_action = st.checkbox("Add Call to Action", value=True)
                with col_c:
                    mention_competitors = st.checkbox("Mention Competitors", value=False)
                
                # Number of portfolio links to include
                num_portfolio_links = st.slider("Number of Portfolio Links to Include", min_value=1, max_value=5, value=2)
        
        with col2:
            submit_button = st.button("Generate Email", type="primary", use_container_width=True)
            
            if submit_button:
                with st.spinner("Analyzing job and generating email..."):
                    try:
                        # Load and process URL
                        loader = WebBaseLoader([url_input])
                        data = clean_text(loader.load().pop().page_content)
                        
                        # Extract job details
                        jobs = chain.extract_jobs(data)
                        
                        if jobs:
                            job = jobs[0]  # Take the first job
                            
                            # Get skills and matching portfolio links
                            skills = job.get('skills', [])
                            links = portfolio.query_links(skills, n_results=num_portfolio_links)
                            
                            # Create personalization options
                            personalization = {
                                "recipient_name": recipient_name,
                                "include_company_research": include_company_research,
                                "company_url": url_input,
                                "add_call_to_action": add_call_to_action,
                                "mention_competitors": mention_competitors
                            }
                            
                            # Generate email
                            email = chain.write_mail(job, links, template_style=email_style, personalization=personalization)
                            
                            # Display job information
                            st.subheader("Job Details")
                            job_details = {
                                "Role": job.get("role", "N/A"),
                                "Company": job.get("company_name", "N/A"),
                                "Experience": job.get("experience", "N/A"),
                                "Skills": ", ".join(job.get("skills", [])) if isinstance(job.get("skills", []), list) else job.get("skills", "N/A")
                            }
                            
                            for key, value in job_details.items():
                                st.write(f"**{key}:** {value}")
                            
                            # Display email
                            st.subheader("Generated Email")
                            st.code(email, language='markdown')
                            
                            # Save button and download
                            col_save, col_followup = st.columns(2)
                            
                            with col_save:
                                if st.button("Save to History"):
                                    save_email_history(email, job, url_input, email_style)
                                    st.success("Email saved to history!")
                            
                            with col_followup:
                                if st.button("Generate Follow-up Email"):
                                    with st.spinner("Creating follow-up..."):
                                        follow_up = chain.generate_follow_up(email)
                                        st.subheader("Follow-up Email")
                                        st.code(follow_up, language='markdown')
                            
                            # Copy to clipboard button (uses JavaScript)
                            st.markdown(
                                """
                                <button 
                                    onclick="navigator.clipboard.writeText(document.querySelector('pre').innerText);alert('Copied to clipboard!')" 
                                    style="background-color:#4CAF50;color:white;padding:10px;border:none;border-radius:5px;cursor:pointer;">
                                    Copy to Clipboard
                                </button>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("No job information found on this page.")
                    except Exception as e:
                        st.error(f"An Error Occurred: {e}")
    
    # Tab 2: Portfolio Manager
    with tab2:
        st.header("Portfolio Manager")
        
        # Display current portfolio
        st.subheader("Current Portfolio Items")
        st.dataframe(portfolio.data)
        
        # Add new portfolio item
        st.subheader("Add New Portfolio Item")
        with st.form("portfolio_form"):
            tech_stack = st.text_input("Technology Stack (comma separated):")
            portfolio_link = st.text_input("Portfolio Link:")
            submit_portfolio = st.form_submit_button("Add to Portfolio")
            
        if submit_portfolio and tech_stack and portfolio_link:
            if portfolio.add_portfolio_item(tech_stack, portfolio_link):
                st.success(f"Added new portfolio item: {tech_stack}")
    
    # Tab 3: Email History
    with tab3:
        st.header("Email History")
        
        history = load_email_history()
        if history:
            for i, entry in enumerate(history):
                with st.expander(f"{entry['date']} - {entry['job_title']} at {entry['company']}"):
                    st.write(f"**URL:** {entry['url']}")
                    st.write(f"**Template:** {entry['template_style']}")
                    st.code(entry['email'], language='markdown')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Generate Follow-up", key=f"followup_{i}"):
                            with st.spinner("Creating follow-up..."):
                                follow_up = chain.generate_follow_up(entry['email'])
                                st.code(follow_up, language='markdown')
        else:
            st.info("No emails in history yet. Generate and save some emails to see them here.")
    
    # Tab 4: Settings
    with tab4:
        st.header("Settings")
        
        # API Key configuration
        api_key = st.text_input("GROQ API Key:", value=os.getenv("GROQ_API_KEY", ""), type="password")
        if st.button("Save API Key"):
            with open(".env", "w") as f:
                f.write(f"GROQ_API_KEY={api_key}")
            st.success("API Key saved!")
        
        # Email templates editor
        st.subheader("Edit Email Templates")
        
        template_type = st.selectbox("Select Template to Edit:", options=list(chain.email_templates.keys()))
        template_text = st.text_area("Template Instructions:", value=chain.email_templates.get(template_type, ""), height=200)
        
        if st.button("Save Template"):
            chain.email_templates[template_type] = template_text
            st.success(f"Template '{template_type}' updated!")


if __name__ == "__main__":
    create_streamlit_app()