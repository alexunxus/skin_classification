
import argparse
import requests
import json
import time
import os
import re
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="File to post to DB")
parser.add_argument("--username", type=str, default="NCKU_skin_admin", help="it is okay to use any account to post")
parser.add_argument("--password", type=str, default="9MBwbuAJ1S", help="it is okay to use any account to post")
parser.add_argument("--API_URL", type=str, default="https://research.aetherai.com/home/viewer/patch_result/add_result/", help="API url to post")
parser.add_argument("--replace_path", type=str, default="True", help="API url to post")
args = parser.parse_args()
URL = "https://research.aetherai.com/home/accounts/login/" #login URL
API = args.API_URL # "https://research.aetherai.com/home/viewer/patch_result/add_result/"
# replace path & file name in json
def replce_path_json(d):
    d['slide_name'] = d['slide_name'].replace('.svs', '') # delete '.svs' if needed
    
    prefix_folder = args.file.split('/')
    prefix_folder = os.path.join('/result', prefix_folder[-4])
    # print('prefix_folder: ', prefix_folder)
    # replace prefix_folder in meta_path
    tmp = d["meta_path"].split('/')
    subject_name = tmp[-2]
    csv = tmp[-1]
    new_path = os.path.join(prefix_folder, subject_name, csv)
    d["meta_path"] = new_path
    # replace folder in every thres path
    for thres in d['layer_map']:
        tmp = d['layer_map'][thres]['path'].split('/')
        subject_name = tmp[-2]
        png = tmp[-1]
        new_path = os.path.join(prefix_folder, subject_name, png)
        d['layer_map'][thres]['path'] = new_path
    
    # with open(args.file, "w") as f:
    #     json.dump(d, f)
    
    return d
with open(args.file) as f:
    d = json.load(f)
if args.replace_path:
    d = replce_path_json(d)
client = requests.session() # create session
# LOGIN with csrftoeken
client.get(URL)  # sets cookie
csrftoken = client.cookies['website-csrf']
login_data = {"username":args.username, 
              "password":args.password, 
              "csrfmiddlewaretoken":csrftoken}
r = client.post(URL, data=login_data, headers=dict(Referer=URL))
csrftoken = client.cookies['website-csrf']
# POST with same csrftoken
cookies = dict(client.cookies)
d["csrfmiddlewaretoken"] = csrftoken
response = requests.post(url=API, data=json.dumps(d), headers={
    'Content-Type': 'application/json',
    "X-CSRFToken": csrftoken,
    'Referer': API
}, cookies=cookies, stream=True, timeout=5)
print(args.file, response)