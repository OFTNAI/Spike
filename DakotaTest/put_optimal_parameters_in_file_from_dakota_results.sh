> "OptimalParameters/$1.txt"
FOO=$(sed -n -e 6,9p dakota_results.txt.txt)
while IFS='' read -r line || [[ -n "$line" ]]; do
	FOO_NO_WHITESPACE="$(echo -e "$line" | tr -d '[[:space:]]')"
	echo $FOO_NO_WHITESPACE >> "OptimalParameters/$1.txt"
done <<< "$FOO"