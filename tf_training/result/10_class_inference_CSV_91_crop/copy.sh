for f in *
do
	TMP_PATH="/mnt/ai_result/research/A19001_NCKU_SKIN/$f/"
	if [ -d "$TMP_PATH" ]; then
		echo "$TMP_PATH exist!"
	else
		echo "$TMP_PATH doesn't exist!"
	fi
	SRC_PATH="$f/mapping.json"
	if [ -f "$SRC_PATH" ]; then
		echo "$SRC_PATH exist!"
		cp "$SRC_PATH" "$TMP_PATH"
	fi
done
