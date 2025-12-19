from django.shortcuts import render, redirect
from .forms import ResumeUploadForm
import re
from PyPDF2 import PdfReader
import docx
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .forms import RegisterForm

def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password']
            )
            login(request, user)
            return redirect("analyzer")
    else:
        form = RegisterForm()

    return render(request, "register.html", {"form": form})

def login_view(request):
    error = None
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("analyzer")
        else:
            error = "Invalid username or password"

    return render(request, "login.html", {"error": error})

def logout_view(request):
    logout(request)
    return redirect("login")


# --------------------------------------------------
# Redirect Home
# --------------------------------------------------
def home_redirect(request):
    return redirect("analyzer")

# --------------------------------------------------
# File Readers
# --------------------------------------------------
def pdf_to_text(file_obj):
    reader = PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def docx_to_text(file_obj):
    document = docx.Document(file_obj)
    return "\n".join([p.text for p in document.paragraphs])

def txt_to_text(file_obj):
    raw = file_obj.read()
    try:
        return raw.decode("utf-8")
    except:
        return raw.decode("latin-1", errors="ignore")

# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------
def clean_text(txt):
    txt = re.sub(r"\r", "\n", txt)
    txt = re.sub(r"\n{2,}", "\n\n", txt)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"[^\x00-\x7F]+", " ", txt)
    return txt.strip()

# --------------------------------------------------
# Basic Details Extraction
# --------------------------------------------------
import re

def extract_details(text):
    # ---------------- EMAIL ----------------
    email_m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_m.group(0) if email_m else "Not found"

    # ---------------- PHONE ----------------
    phone_m = re.search(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}', text)
    phone = phone_m.group(0) if phone_m else "Not found"

    # ---------------- NAME ----------------
    name = "Not found"
    STOP_WORDS = {
        "name", "email", "phone", "contact",
        "linkedin", "github", "portfolio", "resume",
        "data", "analytics", "analyst", "engineer",
        "developer", "enthusiast", "student", "intern"
    }

    # Clean text for name extraction
    clean_text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    clean_text = re.sub(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}', '', clean_text)
    clean_text = re.sub(r'http\S+', '', clean_text)
    clean_text = re.sub(r'linkedin\S+', '', clean_text, flags=re.I)
    clean_text = re.sub(r'github\S+', '', clean_text, flags=re.I)

    lines = clean_text.splitlines()
    for line in lines:
        words = line.split()
        candidate = []
        for w in words:
            w_clean = re.sub(r'[^A-Za-z]', '', w)
            if not w_clean or w_clean.lower() in STOP_WORDS or len(w_clean) <= 1:
                continue
            if w_clean[0].isupper():
                candidate.append(w_clean)
            else:
                if len(candidate) >= 2:
                    break
                else:
                    candidate = []
        if 2 <= len(candidate) <= 4:
            name = " ".join(candidate)
            break

    # Fallback: first line capitalized words
    if name == "Not found" and lines:
        first_line_words = [w for w in lines[0].split() if w[0].isupper()]
        if len(first_line_words) >= 2:
            name = " ".join(first_line_words[:4])

    # ---------------- EDUCATION ----------------
    education = []

    def extract_year(txt):
        y = re.search(r'(19|20)\d{2}', txt)
        return y.group(0) if y else None

    edu_section = re.search(
        r'education\s*(.*?)(?:skills|experience|projects|certifications|$)',
        text,
        re.I | re.S
    )
    edu_text = edu_section.group(1) if edu_section else text

    DEGREE_PATTERNS = {
        "Bachelor of Technology (B.Tech)": r'\bb\.?\s*tech\b|\bbachelor of technology\b',
        "Bachelor of Engineering (B.E)": r'\bb\.?\s*e\b|\bbachelor of engineering\b',
        "Bachelor of Science (B.Sc)": r'\bb\.?\s*sc\b|\bbachelor of science\b',
        "Master of Technology (M.Tech)": r'\bm\.?\s*tech\b',
        "Master of Science (M.Sc)": r'\bm\.?\s*sc\b',
        "MBA": r'\bmba\b',
        "MCA": r'\bmca\b',
    }

    # Extract degrees
    for degree, pattern in DEGREE_PATTERNS.items():
        for m in re.finditer(pattern, edu_text, re.I):
            block = edu_text[m.start():m.start() + 200]
            year = extract_year(block)
            institute_match = re.search(
                r'([A-Z][A-Za-z\s\.]+(College|University|Institute|School|Academy|Institute of Technology))',
                block
            )
            line = degree
            if institute_match:
                line += f", {institute_match.group(1)}"
            if year:
                line += f" ({year})"
            education.append(line)

    # -------- HSC --------
    hsc_seen = False
    for hsc_match in re.finditer(r'\bhsc\b|higher secondary', edu_text, re.I):
        if hsc_seen:
            continue
        block = edu_text[hsc_match.start():hsc_match.start()+100]
        year = extract_year(block)
        education.append(f"Higher Secondary Certificate (HSC){f' ({year})' if year else ''}")
        hsc_seen = True

    # -------- SSC --------
    ssc_seen = False
    for ssc_match in re.finditer(r'\bssc\b|secondary school', edu_text, re.I):
        if ssc_seen:
            continue
        block = edu_text[ssc_match.start():ssc_match.start()+100]
        year = extract_year(block)
        education.append(f"Secondary School Certificate (SSC){f' ({year})' if year else ''}")
        ssc_seen = True

    # -------- REMOVE ANY DUPLICATES WHILE PRESERVING ORDER --------
    seen = set()
    unique_education = []
    for edu in education:
        if edu not in seen:
            unique_education.append(edu)
            seen.add(edu)
    education = unique_education

    if not education:
        education = ["Not found"]

    return {
        "name": name,
        "email": email,
        "contact": phone,
        "education": education
    }

# --------------------------------------------------
# Skills Extraction
# --------------------------------------------------
COMMON_SKILLS = [
    "python","java","c++","javascript","react","angular","node","django","flask",
    "sql","mysql","mongodb","html","css","aws","docker","kubernetes","ml",
    "machine learning","tensorflow","pytorch","git","linux"
]

def extract_skills(text, top_n=8):
    text = text.lower()
    skills = []
    for skill in COMMON_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            skills.append(skill.title())
    return list(dict.fromkeys(skills))[:top_n]

# --------------------------------------------------
# Projects
# --------------------------------------------------
def extract_projects(text, max_projects=5):
    import re

    # Normalize text
    text_clean = re.sub(r'\n+', ' ', text)
    text_clean = re.sub(r'\s+', ' ', text_clean)

    # Find Projects section
    section_match = re.search(
        r"(projects|academic projects)[\s:]*([\s\S]*?)(?=experience|certification|education|skills|achivement|$)",
        text_clean, re.I
    )
    if not section_match:
        return ["Not found"]

    section_text = section_match.group(2).strip()

    # Split by separators
    raw_projects = re.split(r'(?<=\w)\s*(?:—|:|-)\s*', section_text)

    projects = []
    for entry in raw_projects:
        entry = entry.strip()
        if not entry:
            continue

        words = entry.split()
        title = ' '.join(words[:4])  # first 4 words as project title

        # Detect technologies, ignoring duplicates
        tech_matches = re.findall(
            r'\b(Python|Django|Java|C\+\+|C#|Flask|React|Node\.js|HTML|CSS|JavaScript)\b',
            entry, re.I
        )
        tech_used = ', '.join(list(dict.fromkeys([t.capitalize() for t in tech_matches])))

        if tech_used:
            projects.append(f"{title} ({tech_used})")
        else:
            projects.append(title)

        if len(projects) >= max_projects:
            break

    return projects if projects else ["Not found"]


# --------------------------------------------------
# Experience
# --------------------------------------------------
def extract_experience(text, max_entries=5):
    import re

    experiences = []

    # Normalize text
    text_clean = re.sub(r'\n+', '\n', text)  # keep newlines
    text_clean = re.sub(r'\s{2,}', ' ', text_clean)

    # Try to locate Experience section
    match = re.search(
        r'(experience|work experience|professional experience)[\s:]*([\s\S]*?)(?=projects|education|skills|certification|achievement|$)',
        text_clean, re.I
    )
    section = match.group(2).strip() if match else text_clean

    # Split by lines and extract roles
    lines = section.split('\n')
    for line in lines:
        line = line.strip()
        if any(keyword.lower() in line.lower() for keyword in ["intern", "engineer", "developer", "analyst", "manager", "trainee"]):
            experiences.append(line)

        if len(experiences) >= max_entries:
            break

    return experiences if experiences else ["Not found"]



# --------------------------------------------------
# Achievements
# --------------------------------------------------
def extract_achievements(text, max_entries=5):
    import re

    # Normalize text
    text_clean = re.sub(r'\n+', '\n', text)

    # Look for Achievements / Certifications section
    match = re.search(
        r'(achievements|certifications|awards)[\s:]*([\s\S]*?)(?=experience|education|projects|skills|$)',
        text_clean, re.I
    )

    if match:
        section = match.group(2)
        lines = re.split(r'\n|;', section)
        achievements = [l.strip() for l in lines if len(l.strip()) > 3]
        return achievements[:max_entries] if achievements else ["Not found"]

    return ["Not found"]


# --------------------------------------------------
# Category Prediction
# --------------------------------------------------
def predict_category(text):
    text = text.lower()
    if any(x in text for x in ["ml", "machine learning", "tensorflow"]):
        return "Data / AI"
    if any(x in text for x in ["django", "flask", "react"]):
        return "Web Development"
    if any(x in text for x in ["docker", "kubernetes", "aws"]):
        return "DevOps"
    return "General IT"

# --------------------------------------------------
# Job Recommendation (FIXED MAJOR BUG)
# --------------------------------------------------
# Define jobs grouped by categories
JOB_SKILL_MAP = {
    "ML Engineer": ["python", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"],
    "Data Scientist": ["python", "pandas", "numpy", "statistics", "sql", "matplotlib", "seaborn"],
    "Backend Developer": ["python", "django", "flask", "rest api", "sql", "postgresql"],
    "Frontend Developer": ["html", "css", "javascript", "react", "vue", "angular"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "django", "flask", "sql"],
    "Software Engineer": ["python", "c++", "java", "algorithms", "data structures"],
    "Banking Analyst": ["excel", "finance", "accounting", "financial modeling", "statistics", "sql"],
    "Investment Analyst": ["finance", "excel", "valuation", "reporting", "risk analysis", "sql"],
    "Marketing Executive": ["marketing", "seo", "content creation", "digital marketing", "social media"],
    "HR Executive": ["communication", "recruitment", "talent acquisition", "interviewing", "ms office"],
    "Consultant": ["analytics", "presentation", "problem solving", "research", "excel"],
    "Product Manager": ["agile", "roadmap", "project management", "jira", "communication"]
}

def job_recommendation(text):
    """
    Recommend all jobs matching skills in the resume text, 
    ranked by number of skills matched.
    """
    text = text.lower()
    job_scores = {}

    # Count matches per job
    for job, skills in JOB_SKILL_MAP.items():
        match_count = sum(1 for skill in skills if skill in text)
        if match_count > 0:
            job_scores[job] = match_count

    # Sort jobs by number of matched skills (descending)
    sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)

    # Return only the job names
    recommended_jobs = [job for job, score in sorted_jobs]

    # If no jobs match, return default
    return recommended_jobs if recommended_jobs else ["Software Engineer"]

# --------------------------------------------------
# Resume Score
# --------------------------------------------------
def calculate_resume_score(skills, projects, experience):
    return min(len(skills)*5 + len(projects)*10 + len(experience)*5, 100)

# --------------------------------------------------
# Resume Tips
# --------------------------------------------------
def generate_resume_tips(skills, projects, experience, education, score):
    tips = []
    if len(skills) < 5:
        tips.append("Add more relevant technical skills.")
    if len(projects) < 2:
        tips.append("Include more real-world projects.")
    if experience == ["Not found"]:
        tips.append("Add internships or work experience.")
    if education == ["Not found"]:
        tips.append("Mention education details clearly.")
    if score < 60:
        tips.append("Overall resume needs improvement.")
    return tips

# --------------------------------------------------
# Main Analyzer View
# --------------------------------------------------
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.conf import settings
import google.generativeai as genai


@login_required(login_url="login")
def analyzer_view(request):
    form = ResumeUploadForm(request.POST or None, request.FILES or None)
    context = {"form": form, "show_results": False}

    if form.is_valid():
        file = request.FILES["resume"]
        name = file.name.lower()

        if name.endswith(".pdf"):
            text = pdf_to_text(file)
        elif name.endswith(".docx"):
            text = docx_to_text(file)
        elif name.endswith(".txt"):
            text = txt_to_text(file)
        else:
            text = ""

        text = clean_text(text)

        info = extract_details(text)
        skills = extract_skills(text)
        projects = extract_projects(text)
        experience = extract_experience(text)
        achievements = extract_achievements(text)
        category = predict_category(text)
        jobs = job_recommendation(text)
        score = calculate_resume_score(skills, projects, experience)
        tips = generate_resume_tips(skills, projects, experience, info["education"], score)

        context.update({
            "show_results": True,
            "name": info["name"],
            "email": info["email"],
            "contact": info["contact"],
            "education": info["education"],
            "skills": skills,
            "projects": projects,
            "experience": experience,
            "achievements": achievements,
            "predicted_category": category,
            "recommended_jobs": jobs,
            "resume_score": score,
            "resume_tips": tips,
        })

    return render(request, "analyzer.html", context)


@login_required(login_url="login")
def chatbot_view(request):
    ai_response = ""
    category = ""
    skills = ""

    if request.method == "POST":
        category = request.POST.get("category", "")
        skills = request.POST.get("skills", "")
        question = request.POST.get("question", "")

        prompt = f"""
You are SkillSync AI Career Assistant.

Candidate Job Category: {category}
Candidate Skills: {skills}

User Question: {question if question else "Suggest top career guidance and companies"}

Rules:
- Be clear and beginner friendly
- Suggest top 5 companies in India
- Give actionable advice
"""

        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        ai_response = model.generate_content(prompt).text

    return render(request, "chatbot.html", {
        "response": ai_response,
        "category": category,
        "skills": skills
    })