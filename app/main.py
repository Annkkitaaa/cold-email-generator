import os
import re
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Set page config first - this must come before any other Streamlit command
st.set_page_config(layout="wide", page_title="Cold Email Generator Pro", page_icon="📧")

load_dotenv()

# Custom purple theme styling - now applied after set_page_config
st.markdown("""
<style>
    /* Primary purple color and accents */
    .stApp {
        background-color: #f9f7ff;
    }
    .stButton>button {
        background-color: #6a0dad;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8a2be2;
        box-shadow: 0 4px 8px rgba(106, 13, 173, 0.2);
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0e6ff;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: #6a0dad;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6a0dad !important;
        color: white !important;
    }
    h1, h2, h3 {
        color: #6a0dad;
    }
    .stMarkdown a {
        color: #8a2be2;
    }
    /* Form elements */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea {
        border-color: #d0bdf4;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus, .stTextArea>div>div>textarea:focus {
        border-color: #6a0dad;
        box-shadow: 0 0 0 1px #6a0dad;
    }
    /* Code block styling */
    pre {
        background-color: #f0e6ff !important;
        border-left: 4px solid #6a0dad !important;
        border-radius: 4px !important;
        padding: 12px !important;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0e6ff;
        color: #6a0dad;
        border-radius: 4px;
    }
    /* Custom button styling */
    .custom-button {
        display: inline-block;
        padding: 10px 20px;
        background-color: #6a0dad;
        color: white;
        text-align: center;
        text-decoration: none;
        font-weight: bold;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        margin: 5px 0;
        transition: all 0.3s;
    }
    .custom-button:hover {
        background-color: #8a2be2;
        box-shadow: 0 4px 8px rgba(106, 13, 173, 0.2);
    }
    /* Success and error messages */
    .success-message {
        background-color: #e8f0fe;
        border-left: 4px solid #6a0dad;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    /* Spinner styling */
    .stSpinner > div > div {
        border-top-color: #6a0dad !important;
    }
    /* Custom card design */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-top: 4px solid #6a0dad;
    }
</style>
""", unsafe_allow_html=True)

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama3-8b-8192"
        )
        # Make sure there are no unescaped {employees} variables in templates
        self.email_templates = {
            "formal": "You are Mohan, a business development executive at AtliQ. Write a formal and professional cold email to the client regarding the job mentioned above describing AtliQ's capability in fulfilling their needs.",
            "conversational": "You are Mohan, a business development executive at AtliQ. Write a friendly and conversational cold email to the client that shows personality while highlighting AtliQ's capabilities for the job above.",
            "problem-solution": "You are Mohan, a business development executive at AtliQ. Write a cold email that identifies specific problems the client might be facing based on the job description, and position AtliQ's solutions as the answer. Mention how our team of experts can help solve their challenges."
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

    def write_mail(self, job, portfolio_links, template_style="formal", personalization=None):
        """
        Generate an email based on job details using a direct approach without complex template variables.
        """
        try:
            # Get basic job info as strings
            job_str = str(job)
            portfolio_str = str(portfolio_links)
            
            # Get info for company research
            company_research_text = ""
            if personalization and personalization.get('include_company_research', False):
                try:
                    company_url = personalization.get('company_url', '')
                    company_name = job.get('company_name', 'Unknown')
                    company_research = self.extract_company_info(company_url, company_name)
                    
                    company_values = ", ".join([str(v) for v in company_research.get('values', [])])
                    company_initiatives = ", ".join([str(i) for i in company_research.get('initiatives', [])])
                    company_pain_points = ", ".join([str(p) for p in company_research.get('pain_points', [])])
                    company_size = str(company_research.get('size', 'Unknown'))
                    
                    company_research_text = f"""
                    Company Values: {company_values}
                    Recent Initiatives: {company_initiatives}
                    Potential Pain Points: {company_pain_points}
                    Company Size: {company_size}
                    """
                except Exception as e:
                    company_research_text = f"No company research available. Error: {str(e)}"
            
            # Get personalization options
            recipient = personalization.get('recipient_name', '') if personalization else ''
            add_cta = personalization.get('add_call_to_action', False) if personalization else False
            mention_competitors = personalization.get('mention_competitors', False) if personalization else False
            
            # Get template style info
            style_info = ""
            if template_style == "formal":
                style_info = "Write a formal and professional cold email."
            elif template_style == "conversational":
                style_info = "Write a friendly and conversational cold email that shows personality."
            elif template_style == "problem-solution":
                style_info = "Write a cold email that identifies specific problems and positions AtliQ as the solution."
            
            # Create one simple prompt instead of using complex templates
            simple_prompt = f"""
            You are Mohan, a business development executive at AtliQ.
            
            Job details: {job_str}
            
            Portfolio links: {portfolio_str}
            
            {company_research_text}
            
            {style_info}
            
            Write a cold email to the client regarding this job. Describe AtliQ's capability in fulfilling their needs.
            
            AtliQ is an AI & Software Consulting company dedicated to facilitating the seamless integration of business processes through automated tools.
            
            {"Address the email to " + recipient if recipient else ""}
            
            {"Include a call to action at the end." if add_cta else ""}
            
            {"Mention how AtliQ compares favorably to competitors." if mention_competitors else ""}
            
            Format with subject line, greeting, body, and signature.
            """
            
            # Use a direct invocation with a simple text prompt
            response = self.llm.invoke(simple_prompt)
            return response.content
            
        except Exception as e:
            return f"Error generating email: {str(e)}"
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
        try:
            res = chain_followup.invoke({"original_email": original_email, "days": days_passed})
            return res.content
        except Exception as e:
            return f"Error generating follow-up email: {str(e)}"

class SimplePortfolio:
    def __init__(self):
        # Use the CSV data directly to avoid file path issues
        self.data = pd.DataFrame({
            "Techstack": [
                "React, Node.js, MongoDB",
                "Angular,.NET, SQL Server",
                "Vue.js, Ruby on Rails, PostgreSQL",
                "Python, Django, MySQL",
                "Java, Spring Boot, Oracle",
                "Flutter, Firebase, GraphQL",
                "WordPress, PHP, MySQL",
                "Magento, PHP, MySQL",
                "React Native, Node.js, MongoDB",
                "iOS, Swift, Core Data",
                "Android, Java, Room Persistence",
                "Kotlin, Android, Firebase",
                "Android TV, Kotlin, Android NDK",
                "iOS, Swift, ARKit",
                "Cross-platform, Xamarin, Azure",
                "Backend, Kotlin, Spring Boot",
                "Frontend, TypeScript, Angular",
                "Full-stack, JavaScript, Express.js",
                "Machine Learning, Python, TensorFlow",
                "DevOps, Jenkins, Docker"
            ],
            "Links": [
                "https://example.com/react-portfolio",
                "https://example.com/angular-portfolio",
                "https://example.com/vue-portfolio",
                "https://example.com/python-portfolio",
                "https://example.com/java-portfolio",
                "https://example.com/flutter-portfolio",
                "https://example.com/wordpress-portfolio",
                "https://example.com/magento-portfolio",
                "https://example.com/react-native-portfolio",
                "https://example.com/ios-portfolio",
                "https://example.com/android-portfolio",
                "https://example.com/kotlin-android-portfolio",
                "https://example.com/android-tv-portfolio",
                "https://example.com/ios-ar-portfolio",
                "https://example.com/xamarin-portfolio",
                "https://example.com/kotlin-backend-portfolio",
                "https://example.com/typescript-frontend-portfolio",
                "https://example.com/full-stack-js-portfolio",
                "https://example.com/ml-python-portfolio",
                "https://example.com/devops-portfolio"
            ]
        })
        
        # Save to a file for future use (will create in current directory)
        self.file_path = "my_portfolio.csv"
        self.data.to_csv(self.file_path, index=False)
    
    def load_portfolio(self):
        # No need to do anything here as we already loaded the data in __init__
        pass
        
    def query_links(self, skills, n_results=2):
        """Simple skill matching without vector database"""
        relevant_links = []
        
        # Convert skills to lowercase for case-insensitive matching
        if isinstance(skills, list):
            skills_lower = [s.lower() for s in skills]
        else:
            skills_lower = [skills.lower()]
        
        # Score each portfolio entry based on skills match
        scores = []
        for idx, row in self.data.iterrows():
            tech_stack = row["Techstack"].lower()
            score = sum(1 for skill in skills_lower if skill in tech_stack)
            scores.append((idx, score))
        
        # Sort by score and get top n results
        top_indices = [idx for idx, score in sorted(scores, key=lambda x: x[1], reverse=True)[:n_results]]
        
        # Get the links for top matches
        for idx in top_indices:
            relevant_links.append({"links": self.data.iloc[idx]["Links"]})
            
        # If no matches, return some default links
        if not relevant_links:
            for idx in range(min(n_results, len(self.data))):
                relevant_links.append({"links": self.data.iloc[idx]["Links"]})
                
        return relevant_links
    
    def add_portfolio_item(self, techstack, link):
        """Add a new portfolio item"""
        # Add to dataframe
        new_row = pd.DataFrame({"Techstack": [techstack], "Links": [link]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self.data.to_csv(self.file_path, index=False)
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
    # Initialize objects
    chain = Chain()
    portfolio = SimplePortfolio()
    portfolio.load_portfolio()
    
    # Title with logo and description
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <h1 style="margin: 0; color: #6a0dad;">📧 Cold Email Generator Pro</h1>
    </div>
    <p style="font-size: 1.1rem; margin-bottom: 2rem; color: #555;">
        Generate personalized cold emails based on job descriptions with AI-powered insights.
    </p>
    """, unsafe_allow_html=True)
    
    # Create tabs with custom styling
    tab1, tab2, tab3, tab4 = st.tabs(["✉️ Generate Email", "📁 Portfolio Manager", "📚 Email History", "⚙️ Settings"])
    
    # Tab 1: Generate Email
    with tab1:
        st.markdown('<h2 style="color: #6a0dad;">Generate New Cold Email</h2>', unsafe_allow_html=True)
        
        # Wrap content in a card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Input form
        col1, col2 = st.columns([3, 1])
        
        with col1:
            url_input = st.text_input("Enter Job URL:", value="https://jobs.nike.com/job/R-33460")
            
            # More advanced options in an expander
            with st.expander("✨ Personalization Options"):
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
            submit_button = st.button("Generate Email ✨", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
            
        if submit_button:
            with st.spinner("🔍 Analyzing job and generating email..."):
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
                        
                        # Display in a card
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        # Display job information
                        st.markdown('<h3 style="color: #6a0dad;">Job Details</h3>', unsafe_allow_html=True)
                        job_details = {
                            "Role": job.get("role", "N/A"),
                            "Company": job.get("company_name", "N/A"),
                            "Experience": job.get("experience", "N/A"),
                            "Skills": ", ".join(job.get("skills", [])) if isinstance(job.get("skills", []), list) else job.get("skills", "N/A")
                        }
                        
                        for key, value in job_details.items():
                            st.write(f"**{key}:** {value}")
                        
                        # Display email
                        st.markdown('<h3 style="color: #6a0dad; margin-top: 20px;">Generated Email</h3>', unsafe_allow_html=True)
                        st.code(email, language='markdown')
                        
                        # Action buttons
                        col_save, col_followup = st.columns(2)
                        
                        with col_save:
                            if st.button("Save to History 💾"):
                                save_email_history(email, job, url_input, email_style)
                                st.markdown('<div class="success-message">Email saved to history!</div>', unsafe_allow_html=True)
                        
                        with col_followup:
                            if st.button("Generate Follow-up Email 🔄"):
                                with st.spinner("Creating follow-up..."):
                                    follow_up = chain.generate_follow_up(email)
                                    st.markdown('<h3 style="color: #6a0dad; margin-top: 20px;">Follow-up Email</h3>', unsafe_allow_html=True)
                                    st.code(follow_up, language='markdown')
                        
                        # Copy to clipboard button with better styling
                        st.markdown(
                            """
                            <button 
                                onclick="navigator.clipboard.writeText(document.querySelector('pre').innerText);alert('Copied to clipboard!')" 
                                class="custom-button">
                                📋 Copy to Clipboard
                            </button>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">No job information found on this page.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-message">An Error Occurred: {e}</div>', unsafe_allow_html=True)
    
    # Tab 2: Portfolio Manager
    with tab2:
        st.markdown('<h2 style="color: #6a0dad;">Portfolio Manager</h2>', unsafe_allow_html=True)
        
        # Display current portfolio with card styling
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #6a0dad;">Current Portfolio Items</h3>', unsafe_allow_html=True)
        st.dataframe(portfolio.data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add new portfolio item
        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #6a0dad;">Add New Portfolio Item</h3>', unsafe_allow_html=True)
        with st.form("portfolio_form"):
            tech_stack = st.text_input("Technology Stack (comma separated):")
            portfolio_link = st.text_input("Portfolio Link:")
            submit_portfolio = st.form_submit_button("Add to Portfolio")
            
        if submit_portfolio and tech_stack and portfolio_link:
            if portfolio.add_portfolio_item(tech_stack, portfolio_link):
                st.markdown('<div class="success-message">Added new portfolio item!</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Email History
    with tab3:
        st.markdown('<h2 style="color: #6a0dad;">Email History</h2>', unsafe_allow_html=True)
        
        history = load_email_history()
        if history:
            for i, entry in enumerate(history):
                with st.expander(f"📅 {entry['date']} - {entry['job_title']} at {entry['company']}"):
                    st.write(f"**URL:** {entry['url']}")
                    st.write(f"**Template:** {entry['template_style']}")
                    st.code(entry['email'], language='markdown')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Generate Follow-up", key=f"followup_{i}"):
                            with st.spinner("Creating follow-up..."):
                                follow_up = chain.generate_follow_up(entry['email'])
                                st.markdown('<h4 style="color: #6a0dad;">Follow-up Email</h4>', unsafe_allow_html=True)
                                st.code(follow_up, language='markdown')
        else:
            st.info("No emails in history yet. Generate and save some emails to see them here.")
    
    # Tab 4: Settings
    with tab4:
        st.markdown('<h2 style="color: #6a0dad;">Settings</h2>', unsafe_allow_html=True)
        
        # API Key configuration
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #6a0dad;">API Configuration</h3>', unsafe_allow_html=True)
        api_key = st.text_input("GROQ API Key:", value=os.getenv("GROQ_API_KEY", ""), type="password")
        if st.button("Save API Key"):
            with open(".env", "w") as f:
                f.write(f"GROQ_API_KEY={api_key}")
            st.markdown('<div class="success-message">API Key saved!</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Email templates editor
        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #6a0dad;">Edit Email Templates</h3>', unsafe_allow_html=True)
        
        template_type = st.selectbox("Select Template to Edit:", options=list(chain.email_templates.keys()))
        template_text = st.text_area("Template Instructions:", value=chain.email_templates.get(template_type, ""), height=200)
        
        if st.button("Save Template"):
            chain.email_templates[template_type] = template_text
            st.markdown('<div class="success-message">Template updated!</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # About section
        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #6a0dad;">About</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p>Cold Email Generator Pro is an AI-powered tool that helps you create personalized cold emails based on job descriptions. 
        Features include:</p>
        <ul>
            <li>AI-generated personalized cold emails</li>
            <li>Multiple email styles and templates</li>
            <li>Automatic company research</li>
            <li>Portfolio management</li>
            <li>Email history tracking</li>
            <li>Follow-up email generation</li>
        </ul>
        <p>Version 2.0 | © 2025 AtliQ Technologies</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    create_streamlit_app()