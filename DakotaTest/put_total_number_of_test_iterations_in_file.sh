> "OptimalParameters/$1.txt"
FOO="$(tail -2 dakota_tabular.dat | head -1 | awk '{print $1;}')"
echo $FOO >> "OptimalParameters/$1.txt"

# FOO=$(tail -2 dakota_tabular.dat | head -1)

# while IFS='' read -r line || [[ -n "$line" ]]; do
# 	FOO_NO_WHITESPACE="$(echo -e "$line" | tr -d '[[:space:]]')"
# 	echo $FOO_NO_WHITESPACE >> "OptimalParameters/$1.txt"
# done <<< "$FOO"