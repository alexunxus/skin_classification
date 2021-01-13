"""post.py
Post result onto website.
"""

import os
import glob
import argparse
import requests
import json
from tqdm import tqdm
from hephaestus.data.communications import FetchWebData, pattern_checker
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", default="https://10.1.0.8")
    parser.add_argument(
        "--url", default="https://10.1.0.8/home/accounts/login/")
    parser.add_argument(
        "--api", default="https://10.1.0.8/home/viewer/patch_result/add_result/")
    parser.add_argument("--username", default="aetherai")
    parser.add_argument("--password", default="mp6vu;4")
    parser.add_argument(
        "--result_dir", default="/workspace/skin/result/healthexpo/")
    parser.add_argument(
        "--replace_source", default="/workspace/skin/result/healthexpo/")
    parser.add_argument(
        "--replace_target", default="/result/A19001_NCKU_SKIN/")
    args = parser.parse_args()

    client = FetchWebData(target_web=args.website,
                          url=args.url,
                          api=args.api,
                          username=args.username,
                          password=args.password)

    # don't collect postdb.py
    valid_file = []
    for f in os.listdir(args.result_dir):
        if not f.endswith(".py"):
            valid_file.append(f)
    json_files = [os.path.join(args.result_dir, i, "mapping.json")
                  for i in valid_file]
    rep_map = {args.replace_source: args.replace_target}
    print(rep_map)

    for json_file in tqdm(json_files):
        with open(json_file, "r") as f:
            x = json.load(f)
        try:
            resp = client.throw_patch_annotation(x, replaced_path=rep_map)
            print(resp)
        except:
            print("File {} failed to post".format(json_file))
