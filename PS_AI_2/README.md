## How It Works

### Step 1: Load Documents
The system reads all policy text files.

### Step 2: Sentence Splitting
Each document is split into individual sentences.
Each sentence is stored along with its document name.

### Step 3: TF-IDF Vectorization
All sentences are converted into numerical vectors using TF-IDF.

### Step 4: User Query Processing
When a user enters a question:
- The query is vectorized
- Cosine similarity is computed
- Most relevant sentence is selected

### Step 5: Threshold Check
If similarity score is below threshold:
- Return: "Information not available in policy documents."

Otherwise:
- Return best matching sentence
- Display source document name

---

## How To Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run command
```bash
streamlit run app.py
```