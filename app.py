import math

# Define skill aliases
SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r", "kotlin": "kotlin",
    "machinelearning": "machinelearning", "machine learning": "machinelearning",
    "ml": "machinelearning", "sklearn": "machinelearning",
    "deeplearning": "deeplearning", "deep learning": "deeplearning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "datavisualization", "data visualization": "datavisualization",
    "data viz": "datavisualization", "matplotlib": "datavisualization",
    "tableau": "datavisualization", "power-bi": "datavisualization",
    "power bi": "datavisualization", "powerbi": "datavisualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "htmlcss", "html css": "htmlcss",
    "html": "htmlcss", "css": "htmlcss",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "springboot", "springboot": "springboot",
    "rest api": "restapi", "rest": "restapi", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "cicd", "cicd": "cicd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "datastructures", "data structures": "datastructures",
    "competitive programming": "competitive_programming",
    "ui/ux": "uiux", "ui ux": "uiux", "figma": "figma",
}


#Define resume data
resumes_raw = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]



#Normalize and deduplicate skills
def normalizeskills(rawskills_str):
    # Sort aliases by length descending (match multi-word phrases first)
    sortedaliases = sorted(SKILLALIASES.keys(), key=len, reverse=True)
    tokens = [t.strip().lower() for t in rawskillsstr.split(",")]
    canonical = []
    seen = set()
    for token in tokens:
        matched = False
        for alias in sorted_aliases:
            if token == alias:
                canon = SKILL_ALIASES[alias]
                if canon not in seen:
                    canonical.append(canon)
                    seen.add(canon)
                matched = True
                break
        # discard if not matched
    return canonical

print("=== STEP 1+2: Normalized & Deduplicated Skills ===")
resumes = []
for name, raw in resumes_raw:
    skills = normalize_skills(raw)
    resumes.append((name, skills))
    print(f"{name}: {skills}")


#Build vocabulary
all_skills = set()
for _, skills in resumes:
    all_skills.update(skills)
vocab = sorted(all_skills)
vocab_index = {s: i for i, s in enumerate(vocab)}
V = len(vocab)

print(f"\n=== STEP 3: Vocabulary ({V} skills) ===")
print(vocab)


Calculate TF-IDF vectors
N_docs = len(resumes)

Document frequency
df = {skill: 0 for skill in vocab}
for _, skills in resumes:
    for s in skills:
        df[s] += 1

IDF
idf = {skill: math.log(N_docs / df[skill]) for skill in vocab}

print("\n=== IDF values ===")
for s in vocab:
    print(f"  {s}: df={df[s]}, idf={idf[s]:.6f}")

# TF-IDF vectors
tfidf_vectors = []
for name, skills in resumes:
    N = len(skills)
    vec = [0.0] * V
    for s in skills:
        idx = vocab_index[s]
        tf = 1.0 / N
        vec[idx] = tf * idf[s]
    tfidf_vectors.append((name, vec))

print("\n=== STEP 4: TF-IDF Vectors (non-zero only) ===")
for name, vec in tfidf_vectors:
    nonzero = [(vocab[i], round(vec[i], 6)) for i in range(V) if vec[i] > 0]
    print(f"{name}: {nonzero}")

Define job descriptions
jd_raw = [
    ("JD-1 Kakao ML Engineer",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics"),
    ("JD-2 Naver Backend Engineer",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis"),
    ("JD-3 Line Frontend Engineer",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS"),
]

#Build job description binary vectors
def buildjdvector(jdskillsstr):
    vec = [0] * V
    tokens = [t.strip().lower() for t in jdskillsstr.split(",")]
    sortedaliases = sorted(SKILLALIASES.keys(), key=len, reverse=True)
    for token in tokens:
        for alias in sorted_aliases:
            if token == alias:
                canon = SKILL_ALIASES[alias]
                if canon in vocab_index:
                    vec[vocab_index[canon]] = 1
                break
    return vec

jd_vectors = []
print("\n=== STEP 5: JD Binary Vectors (skills matched in vocab) ===")
for jdname, jdrawstr in jdraw:
    vec = buildjdvector(jdrawstr)
    matched = [vocab[i] for i in range(V) if vec[i] == 1]
    jdvectors.append((jdname, vec))
    print(f"{jd_name}: {matched}")