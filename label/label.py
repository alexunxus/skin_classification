import os, json, requests
import json
from tqdm.notebook import tqdm
from datetime import datetime
from hephaestus.data.communications import FetchWebData
from requests.exceptions import ConnectionError

## Settings for the viewer
target_web = "research.aetherai.com"
URL = f"https://{target_web}/home/accounts/login/"
API = f"https://{target_web}/home/viewer/slide/"
username = "NCKU_skin_admin"
password = "9MBwbuAJ1S"

web_obj = FetchWebData(target_web=target_web,
                       url=URL,
                       api=API,
                       username=username,
                       password=password,
                       annot_type="contour_result")
target_dict = web_obj.fetch_target_dict(page_size=80)
print(target_dict.keys())

target_ai_project = "SKIN"
annotation_dict = {}

for key in tqdm(target_dict.keys()):
    # if not key.startswith("2019"): continue
    uuid = target_dict[key]
    ai_project_dict = web_obj.fetch_ai_project_pk_of_slide(uuid)
    if target_ai_project not in ai_project_dict: continue
    ai_pk = ai_project_dict[target_ai_project]
    try:
        data = web_obj.fetch_contour_annotation(uuid, ai_pk)
        if not data:
            annotation_dict[key] = []
        else:
            annotation_dict[key] = data
    except ConnectionError:
        print(f"connection error: {key}")
        continue
print(annotation_dict.keys())
#print(annotation_dict["2019"].keys())
#print(len(annotation_dict["2019"]["targets"]))

import json
with open('label.json', 'w') as fp:
    json.dump(annotation_dict, fp)