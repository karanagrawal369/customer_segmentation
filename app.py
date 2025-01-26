from fastapi import FastAPI, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64


app = FastAPI()


templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main HTML page.
    """
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/cluster", response_class=HTMLResponse)
async def cluster(request: Request, clusters: int = Form(...), file: UploadFile = None):
    """
    Perform K-Means clustering and display results.
    """
    if file is not None:
        content = await file.read()
        data = pd.read_csv(io.BytesIO(content))

        # Ensure required columns exist
        if "Annual Income (k$)" not in data.columns or "Spending Score (1-100)" not in data.columns:
            return HTMLResponse("Error: Dataset must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

        # Extract relevant features
        X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
        labels = kmeans.labels_

        # Create cluster visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis")
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="red")
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        plt.title(f"K-Means Clustering with {clusters} Clusters")

        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()

        # Render results
        return HTMLResponse(f"""
        <html>
            <body>
                <h1>Clustering Results</h1>
                <img src="data:image/png;base64,{img_str}" alt="Cluster Plot">
                <br><br>
                <a href="/">Go Back</a>
            </body>
        </html>
        """)

    return HTMLResponse("Error: Please upload a valid CSV file.")
