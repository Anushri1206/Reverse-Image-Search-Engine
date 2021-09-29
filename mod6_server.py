import numpy as np
from PIL import Image
import import_ipynb
from feature_extractor import FeatureExtractorResnet
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from annoy import AnnoyIndex
import csv

app = Flask(__name__)


fe = FeatureExtractorResnet()
features = []
img_paths = []
img_id = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
    img_id.append(feature_path.stem)

# feature_dim = len(features[0])
# ann_index = AnnoyIndex(feature_dim, metric='angular')
# for i in range(len(features)):
#     ann_index.add_item(i, features[i])

# ann_index.build(100)

ann_index = AnnoyIndex(len(features[0]),metric = 'angular')
ann_index.load('image_indexing_final')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        N = request.form.get('number_images',None)
        print(N)
        N = int(N)
        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        feature = fe.extract(img)
        query = np.array(feature)
        res = ann_index.get_nns_by_vector(query, N ,include_distances=True)
        scores = []
        links = []
        imagename = []
        for i in range(len(res[0])):
            scores.append([res[1][i],img_paths[res[0][i]]])
            imagename.append(img_id[res[0][i]])
            print(uploaded_img_path)

        print(imagename)
        for i in range(N):
            with open("combined_data.csv") as f:
                reader = csv.reader(f)
                query = imagename[i]
                print(scores[i][1])
                for row in reader:                   
                    if(row[1]==query):
                        links.append(row[3])
                        print(row[0])
                        print(row[3])
                        break
        print(links)
        results = []
        for i in range (len(scores)):
            results.append([scores[i][0],scores[i][1],links[i]])
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               results= results)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")