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

load_dotenv()

# Set up page configuration with web3 theme
st.set_page_config(
    layout="wide", 
    page_title="Web3 Cold Email Generator", 
    page_icon="🚀"
)

# Apply custom Web3 styling with purple theme
web3_theme = """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #8A2BE2;
        --secondary-color: #7B68EE;
        --accent-color: #9370DB;
        --background-color: #0E0B16;
        --text-color: #E7DFDD;
        --card-color: #201C2B;
        --card-border: rgba(138, 43, 226, 0.5);
        --success-color: #4CAF50;
        --warning-color: #FFC107;
    }
    
    /* Override Streamlit main elements */
    .stApp {
        background: linear-gradient(135deg, var(--background-color) 0%, #1A1A2E 100%);
    }
    .stTextInput > div > div > input {
        background-color: rgba(30, 30, 30, 0.7);
        color: var(--text-color);
        border: 1px solid var(--primary-color);
        border-radius: 8px;
    }
    .stTextInput > label, .stSelectbox > label, .stSlider > label {
        color: var(--accent-color) !important;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(138, 43, 226, 0.5);
    }
    
    /* Headers with Web3 styling */
    h1, h2, h3 {
        color: var(--text-color) !important;
        text-shadow: 0 0 10px rgba(138, 43, 226, 0.5);
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    h1 {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em !important;
    }
    
    /* Custom card styling */
    .web3-card {
        background-color: var(--card-color);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(138, 43, 226, 0.3);
        transition: all 0.3s ease;
    }
    .web3-card:hover {
        box-shadow: 0 8px 30px rgba(138, 43, 226, 0.5);
        transform: translateY(-3px);
    }
    
    /* Custom expander styling */
    .streamlit-expanderHeader {
        background-color: var(--card-color);
        color: var(--text-color) !important;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: var(--card-color) !important;
        color: var(--text-color) !important;
        border-radius: 8px !important;
    }
    .dataframe th {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Code display styling */
    pre {
        background-color: rgba(30, 30, 40, 0.7) !important;
        border-left: 4px solid var(--primary-color) !important;
        border-radius: 8px !important;
    }
    
    /* Web3 decorative elements */
    .web3-icon {
        display: inline-block;
        padding: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        margin-right: 10px;
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.5);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-color);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        border: 1px solid var(--card-border);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
</style>
"""

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
        self.email_templates = {
            "formal": "You are Mohan, a business development executive at AtliQ. Write a formal and professional cold email to the client regarding the job mentioned above describing AtliQ's capability in fulfilling their needs.",
            "conversational": "You are Mohan, a business development executive at AtliQ. Write a friendly and conversational cold email to the client that shows personality while highlighting AtliQ's capabilities for the job above.",
            "problem-solution": "You are Mohan, a business development executive at AtliQ. Write a cold email that identifies specific problems the client might be facing based on the job description, and position AtliQ's solutions as the answer.",
            "web3-focused": "You are Mohan, a business development executive at AtliQ. Write a professional email highlighting AtliQ's Web3 expertise and blockchain capabilities relevant to the job description. Reference any relevant Web3 projects in the portfolio and explain how AtliQ can leverage blockchain technology to solve their challenges."
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
                5. Any Web3 or blockchain involvement
                
                Format the response as JSON with these keys: values, initiatives, pain_points, size, web3_involvement.
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
                return {"values": [], "initiatives": [], "pain_points": [], "size": "Unknown", "web3_involvement": "None detected"}
                
        except Exception as e:
            print(f"Error extracting company info: {e}")
            return {"values": [], "initiatives": [], "pain_points": [], "size": "Unknown", "web3_involvement": "None detected"}

    def write_mail(self, job, portfolio_links, template_style="formal", personalization=None):
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
                Web3 Involvement: {company_research.get('web3_involvement', 'None detected')}
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
            
            Add the most relevant ones from the following links to showcase Atliq's portfolio: {{portfolio_links}}
            
            Remember you are Mohan, BDE at AtliQ.
            Format the email properly with subject line, greeting, body, and signature.
            
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "portfolio_links": portfolio_links})
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

class SimplePortfolio:
    def __init__(self):
        # Use the CSV data directly to avoid file path issues
        self.data = pd.DataFrame({
            "Techstack": [
                "React, Node.js, MongoDB, Ethereum",
                "Angular, .NET, SQL Server, Solidity",
                "Vue.js, Ruby on Rails, PostgreSQL, Web3.js",
                "Python, Django, MySQL, Blockchain API",
                "Java, Spring Boot, Oracle, NFT Marketplace",
                "Flutter, Firebase, GraphQL, Smart Contracts",
                "WordPress, PHP, MySQL, Cryptocurrency Integration",
                "Magento, PHP, MySQL, DeFi Solutions",
                "React Native, Node.js, MongoDB, DApp Development",
                "iOS, Swift, Core Data, Wallet Integration",
                "Android, Java, Room Persistence, Blockchain Analytics",
                "Kotlin, Android, Firebase, Token Development",
                "Android TV, Kotlin, Android NDK, Metaverse",
                "iOS, Swift, ARKit, Blockchain Security",
                "Cross-platform, Xamarin, Azure, DAO Tools",
                "Backend, Kotlin, Spring Boot, Lightning Network",
                "Frontend, TypeScript, Angular, Zero-Knowledge Proofs",
                "Full-stack, JavaScript, Express.js, Polygon Network",
                "Machine Learning, Python, TensorFlow, Web3 Analytics",
                "DevOps, Jenkins, Docker, Hyperledger"
            ],
            "Links": [
                "https://example.com/ethereum-portfolio",
                "https://example.com/solidity-portfolio",
                "https://example.com/web3js-portfolio",
                "https://example.com/blockchain-api-portfolio",
                "https://example.com/nft-marketplace-portfolio",
                "https://example.com/smart-contracts-portfolio",
                "https://example.com/crypto-integration-portfolio",
                "https://example.com/defi-solutions-portfolio",
                "https://example.com/dapp-development-portfolio",
                "https://example.com/wallet-integration-portfolio",
                "https://example.com/blockchain-analytics-portfolio",
                "https://example.com/token-development-portfolio",
                "https://example.com/metaverse-portfolio",
                "https://example.com/blockchain-security-portfolio",
                "https://example.com/dao-tools-portfolio",
                "https://example.com/lightning-network-portfolio",
                "https://example.com/zero-knowledge-proofs-portfolio",
                "https://example.com/polygon-network-portfolio",
                "https://example.com/web3-analytics-portfolio",
                "https://example.com/hyperledger-portfolio"
            ]
        })
        
        # Save to a file for future use (will create in current directory)
        self.file_path = "web3_portfolio.csv"
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
        
        # Add web3 related terms to always consider
        web3_terms = ["web3", "blockchain", "crypto", "defi", "nft", "token", "ethereum", "bitcoin", "smart contract"]
        
        # Score each portfolio entry based on skills match
        scores = []
        for idx, row in self.data.iterrows():
            tech_stack = row["Techstack"].lower()
            
            # Calculate base score from skills match
            base_score = sum(1 for skill in skills_lower if skill in tech_stack)
            
            # Add bonus for web3 terms
            web3_bonus = sum(2 for term in web3_terms if term in tech_stack)
            
            # Combine scores
            total_score = base_score + web3_bonus
            scores.append((idx, total_score))
        
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
    # Apply the Web3 theme
    st.markdown(web3_theme, unsafe_allow_html=True)
    
    # Initialize objects
    chain = Chain()
    portfolio = SimplePortfolio()
    portfolio.load_portfolio()
    
    # Add Web3 decorative elements to title
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div class="web3-icon">🚀</div>
        <h1>Web3 Cold Email Generator Pro</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Animated subtitle
    st.markdown("""
    <p style="color: #9370DB; font-size: 1.2em; font-style: italic; margin-bottom: 2rem; text-shadow: 0 0 5px rgba(147, 112, 219, 0.5);">
        Empowering outreach with next-gen blockchain technology
    </p>
    """, unsafe_allow_html=True)
    
    # Create tabs with custom styling
    tab1, tab2, tab3, tab4 = st.tabs(["✉️ Generate Email", "📂 Portfolio Manager", "📚 Email History", "⚙️ Settings"])
    
    # Tab 1: Generate Email
    with tab1:
        st.markdown('<h2>Generate New Cold Email</h2>', unsafe_allow_html=True)
        
        # Wrap the form in a Web3 card
        st.markdown('<div class="web3-card">', unsafe_allow_html=True)
        
        # Input form
        col1, col2 = st.columns([3, 1])
        
        with col1:
            url_input = st.text_input("Enter Job URL:", value="https://jobs.nike.com/job/R-33460")
            
            # More advanced options in an expander
            with st.expander("✨ Personalization Options"):
                email_style = st.selectbox(
                    "Email Style",
                    options=["formal", "conversational", "problem-solution", "web3-focused"],
                    index=3
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
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.button("🚀 Generate Email", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
            
        if submit_button:
            with st.spinner("🔮 Analyzing job and generating Web3 email..."):
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
                        
                        # Display job information in a fancy card
                        st.markdown('<div class="web3-card">', unsafe_allow_html=True)
                        st.markdown('<h3>🎯 Job Details</h3>', unsafe_allow_html=True)
                        
                        job_details = {
                            "Role": job.get("role", "N/A"),
                            "Company": job.get("company_name", "N/A"),
                            "Experience": job.get("experience", "N/A"),
                            "Skills": ", ".join(job.get("skills", [])) if isinstance(job.get("skills", []), list) else job.get("skills", "N/A")
                        }
                        
                        # Generate more Web3-styled job details display
                        for key, value in job_details.items():
                            st.markdown(f"""
                            <div style="margin-bottom: 10px;">
                                <span style="color: var(--primary-color); font-weight: bold;">{key}:</span> 
                                <span style="color: var(--text-color);">{value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display email in a fancy card
                        st.markdown('<div class="web3-card">', unsafe_allow_html=True)
                        st.markdown('<h3>📧 Generated Email</h3>', unsafe_allow_html=True)
                        st.code(email, language='markdown')
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Action buttons in a card
                        st.markdown('<div class="web3-card">', unsafe_allow_html=True)
                        st.markdown('<h3>🎬 Actions</h3>', unsafe_allow_html=True)
                        
                        col_save, col_followup = st.columns(2)
                        
                        with col_save:
                            if st.button("💾 Save to History", use_container_width=True):
                                save_email_history(email, job, url_input, email_style)
                                st.success("Email saved to history!")
                        
                        with col_followup:
                            if st.button("🔄 Generate Follow-up", use_container_width=True):
                                with st.spinner("Creating follow-up..."):
                                    follow_up = chain.generate_follow_up(email)
                                    st.subheader("Follow-up Email")
                                    st.code(follow_up, language='markdown')
                        
                        # Copy to clipboard button with Web3 styling
                        st.markdown(
                            """
                            <button 
                                onclick="navigator.clipboard.writeText(document.querySelector('pre').innerText);alert('Copied to clipboard!');" 
                                style="background: linear-gradient(90deg, #8A2BE2, #7B68EE); 
                                      color: white; 
                                      padding: 10px 20px; 
                                      border: none; 
                                      border-radius: 8px; 
                                      margin-top: 10px;
                                      cursor: pointer;
                                      width: 100%;
                                      font-weight: bold;
                                      transition: all 0.3s ease;"
                                onmouseover="this.style.transform='scale(1.02)'"
                                onmouseout="this.style.transform='scale(1)'"
                                >
                                📋 Copy to Clipboard
                            </button>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    else:
                        st.error("No job information found on this page.")
                except Exception as e:
                    st.error(f"An Error Occurred: {e}")
    
    # Tab 2: Portfolio Manager
    with tab2:
        st.markdown('<h2>Portfolio Manager</h2>', unsafe_allow_html=True)
        
        # Display current portfolio in a Web3 card
        st.markdown('<div class="web3-card">', unsafe_allow_html=True)
        st.markdown('<h3>📊 Current Portfolio Items</h3>', unsafe_allow_html=True)
        st.dataframe(portfolio.data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add new portfolio item in a Web3 card
        st.markdown('<div class="web3-card">', unsafe_allow_html=True)
        st.markdown('<h3>➕ Add New Portfolio Item</h3>', unsafe_allow_html=True)
        
        with st.form("portfolio_form"):
            tech_stack = st.text_input("Technology Stack (comma separated):")
            portfolio_link = st.text_input("Portfolio Link:")
            submit_portfolio = st.form_submit_button("Add to Portfolio")
            
        if submit_portfolio and tech_stack and portfolio_link:
            if portfolio.add_portfolio_item(tech_stack, portfolio_link):
                st.success(f"Added new portfolio item: {tech_stack}")
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Email History
    with tab3:
        st.markdown('<h2>Email History</h2>', unsafe_allow_html=True)
        
        history = load_email_history()
        if history:
            st.markdown('<div class="web3-card">', unsafe_allow_html=True)
            
            for i, entry in enumerate(history):
                with st.expander(f"📅 {entry['date']} - {entry['job_title']} at {entry['company']}"):
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <span style="color: var(--primary-color); font-weight: bold;">URL:</span> 
                        <span style="color: var(--text-color);">{entry['url']}</span>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span style="color: var(--primary-color); font-weight: bold;">Template:</span> 
                        <span style="color: var(--text-color);">{entry['template_style']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code(entry['email'], language='markdown')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔄 Generate Follow-up", key=f"followup_{i}", use_container_width=True):
                            with st.spinner("Creating follow-up..."):
                                follow_up = chain.generate_follow_up(entry['email'])
                                st.code(follow_up, language='markdown')
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("💫 No emails in history yet. Generate and save some emails to see them here.")