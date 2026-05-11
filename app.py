import math
# from collections import defaultdict

# 1. DATASETS
candidates = [
    {"id": "01", "name": "Arjun Sharma", "skills": "Pyhton, Machine Learning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "name": "Priya Nair", "skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "name": "Rahul Gupta", "skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "name": "Sneha Patel", "skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "name": "Vikram Singh", "skills": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "name": "Ananya Krishnan", "skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "name": "Karan Mehta", "skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "name": "Deepika Rao", "skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "name": "Aditya Kumar", "skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "name": "Meera lyer", "skills": "python, R, statistics, ML, regression, clustering, Power-Bl"}
]

jds = [
    {"id": "JD-1", "title": "Kakao (ML Engineer)", "req": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization", "pref": "NLP, BERT, Feature Engineering, Statistics"},
    {"id": "JD-2", "title": "Naver (Backend Engineer)", "req": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes", "pref": "REST API, CI/CD, Redis"},
    {"id": "JD-3", "title": "Line (Frontend Engineer)", "req": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS", "pref": "Node.js, GraphQL, Redux, Jest, AWS"}
]



# 2. SKILL ALIASES 
# Fixed minor OCR missing components based strictly on canonical mapping
SKILL_ALIASES = {
    "python": "python", "pyhton": "python", "java": "java", "javascript": "javascript",
    "javascrpit": "javascript", "js": "javascript", "typescript": "typescript",
    "typescrpit": "typescript", "c++": "cpp", "cpp": "cpp", "r": "r", 
    "kotlin": "kotlin", "machinelearning": "machine_learning",
    "machine learning": "machine_learning", "ml": "machine_learning",
    "sklearn": "machine_learning", "deeplearning": "deep_learning",
    "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nip": "nlp", "nlp": "nlp", "bert": "bert", "xgboost": "xgboost", 
    "feature engineering": "feature_engineering", "statistics": "statistics", 
    "stats": "statistics", "regression": "regression", "clustering": "clustering", 
    "data-viz": "data_visualization", "data visualization": "data_visualization", 
    "data viz": "data_visualization", "matplotlib": "data_visualization", 
    "tableau": "data_visualization", "power-bi": "data_visualization", 
    "power bi": "data_visualization", "powerbi": "data_visualization", 
    "pandas": "pandas", "numpy": "numpy", "react": "react", "reacts": "react", 
    "reactjs": "react", "vue": "vue", "vue.js": "vue", "vuejs": "vue", 
    "redux": "redux", "tailwind": "tailwind", "html/css": "html_css", 
    "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql", "node.js": "nodejs", "nodejs": "nodejs",
    "node js": "nodejs", "flask": "flask", "spring boot": "spring_boot",
    "springboot": "spring_boot", "rest api": "rest_api", "rest": "rest_api",
    "restapi": "rest_api", "microservices": "microservices", "sql": "sql",
    "mysql": "mysql", "mysq": "mysql", "postgresql": "postgresql",
    "postgres": "postgresql", "mongodb": "mongodb", "redis": "redis",
    "docker": "docker", "kubernetes": "kubernetes", "kubernates": "kubernetes",
    "k8s": "kubernetes", "ci/cd": "ci_cd", "cicd": "ci_cd", "ci ca": "ci_cd",
    "aws": "aws", "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma"
}

# 3. HELPER FUNCTIONS
def normalize_and_dedupe(skill_string):
    """Splits by comma, lowercases, maps to alias, deduplicates canonicals."""
    raw_skills = [s.strip().lower() for s in skill_string.split(",")]
    canonical_skills = set()
    for skill in raw_skills:
        if skill in SKILL_ALIASES:
            canonical_skills.add(SKILL_ALIASES[skill])
    return sorted(list(canonical_skills))

def vector_norm(vector):
    """Computes Euclidean norm of a vector array."""
    return math.sqrt(sum(v**2 for v in vector))

def dot_product(v1, v2):
    """Computes dot product of two vector arrays."""
    return sum(a * b for a, b in zip(v1, v2))

# 4. PROCESSING RESUMES
resume_skills = {}
skill_document_frequency = defaultdict(int)

# Normalize and calculate Document Frequency (df)
for c in candidates:
    n_skills = normalize_and_dedupe(c["skills"])
    resume_skills[c["id"]] = n_skills
    for skill in n_skills:
        skill_document_frequency[skill] += 1

# 5. BUILD VOCABULARY
# "Create a shared vocabulary from normalized, deduplicated resume skills only"
vocabulary = sorted(list(skill_document_frequency.keys()))
total_resumes = len(candidates)

# Compute Resume TF-IDF Vectors
resume_vectors = {}
for c in candidates:
    c_id = c["id"]
    r_skills = resume_skills[c_id]
    N = len(r_skills)
    
    vec = []
    for vocab_skill in vocabulary:
        if vocab_skill in r_skills:
            tf = 1.0 / N
            idf = math.log(total_resumes / skill_document_frequency[vocab_skill])
            vec.append(tf * idf)
        else:
            vec.append(0.0)
    resume_vectors[c_id] = vec

# 6. PROCESS JDS & CALCULATE MATCHES
results = {}

for jd in jds:
    # Combine Req and Pref skills
    jd_raw_skills = jd["req"] + ", " + jd["pref"]
    jd_normalized = normalize_and_dedupe(jd_raw_skills)
    
    # Build JD Binary Vector based ONLY on shared vocabulary
    jd_vector = []
    for vocab_skill in vocabulary:
        if vocab_skill in jd_normalized:
            jd_vector.append(1.0)
        else:
            jd_vector.append(0.0)
            
    jd_norm = vector_norm(jd_vector)
    
    jd_scores = []
    for c in candidates:
        c_id = c["id"]
        r_vec = resume_vectors[c_id]
        
        r_norm = vector_norm(r_vec)
        
        if r_norm == 0 or jd_norm == 0:
            score = 0.0
        else:
            score = dot_product(r_vec, jd_vector) / (r_norm * jd_norm)
            
        jd_scores.append({
            "name": c["name"],
            "score": round(score, 2)
        })
        
    # Sort by score descending, then by name alphabetically (tiebreaker)
    jd_scores.sort(key=lambda x: (-x["score"], x["name"]))
    
    # Get top 3
    top_3 = jd_scores[:3]
    results[jd["id"]] = f"{top_3[0]['name']} ({top_3[0]['score']:.2f}), {top_3[1]['name']} ({top_3[1]['score']:.2f}), {top_3[2]['name']} ({top_3[2]['score']:.2f})"

# 7. OUTPUT
print("JD-1 Result:", results["JD-1"])
print("JD-2 Result:", results["JD-2"])
print("JD-3 Result:", results["JD-3"])



