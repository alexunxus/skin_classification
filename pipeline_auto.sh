python3 inference.py
./fast_recolor.sh
cp -r /workspace/skin/result/inference_CSV/* /mnt/ai_result/research/A19001_NCKU_SKIN/
cd /mnt/ai_result/research/A19001_NCKU_SKIN/
python postdb.py
