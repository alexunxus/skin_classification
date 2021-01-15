"""
==============postdb.py==================
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
    parser.add_argument("--website", default="research.aetherai.com")
    parser.add_argument(
        "--url", default="https://research.aetherai.com/home/accounts/login/")
    parser.add_argument(
        "--api", default="https://research.aetherai.com/home/viewer/patch_result/add_result/")
    parser.add_argument("--username", default="cgmhpath_npc")
    parser.add_argument("--password", default="HUIF4752zqvm")
    parser.add_argument(
        "--result_dir", default="/workspace/skin/first_stage_inference/inference_result/tf/" )#"/mnt/ai_result/research/A19001_NCKU_SKIN/")
    parser.add_argument(
        "--replace_source", default="/workspace/skin/first_stage_inference/inference_result/tf/")#"/workspace/skin/result/inference_CSV/")
    parser.add_argument(
        "--replace_target", default="/result/A19001_NCKU_SKIN/")
    args = parser.parse_args()

    client = FetchWebData(target_web=args.website,
                          url=args.url,
                          api=args.api,
                          username=args.username,
                          password=args.password)

    json_files = [os.path.join(args.result_dir, i, "mapping.json")
                  for i in os.listdir(args.result_dir) if len(os.listdir(os.path.join(args.result_dir, i)))]
    rep_map = {args.replace_source: args.replace_target}
    print(rep_map)

    for json_file in tqdm(json_files):
        with open(json_file, "r") as f:
            x = json.load(f)
        # temporally
        try:
            client.throw_patch_annotation(x, replaced_path=rep_map)
        except:
            print("File {} failed to post".format(json_file))
