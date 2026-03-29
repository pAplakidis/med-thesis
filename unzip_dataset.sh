#!/bin/bash
find data/ -type f -name "*.zip" -print0 | while IFS= read -r -d '' zip; do
  dir=$(dirname "$zip")
  tmp=$(mktemp -d)

  unzip -o "$zip" -d "$tmp" >/dev/null

  for f in "$tmp"/*; do
    base=$(basename "$f")
    target="$dir/$base"

    if [ -e "$target" ]; then
      i=1
      name="${base%.json}"
      ext="${base##*.}"
      while [ -e "$dir/${name}_$i.$ext" ]; do
        ((i++))
      done
      mv "$f" "$dir/${name}_$i.$ext"
    else
      mv "$f" "$target"
    fi
  done

  rm -rf "$tmp"
done
